"""

"""




import torch
import torch.nn as nn
from typing import *
# einops: 텐서(의 사이즈)를 조금 더 내 자유자재로 만들 수 있는 라이브러리
from einops import rearrange 
from einops.layers.torch import Rearrange

## Helpers
def pair(t: int) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


# patch dropout
# TODO: Understand
class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

## For Vit Transfomer Encoder
class VitTransformerEncoder(nn.Module):
    '''ViT Transformer Encoder differs merely from the original
    [1. Layer Normlization]
    [2. Multi-Head self Attention Layer]
        - V & K & Q
        - Linear(V, K, Q)
        ---- Scaled Dot-Product Attention ----
        - Q & K MatMul
        - (X) Scale
        - (X) Mask (option)
        - result = SoftMax
        - out = result & v MatMul
        ----            Out               ----
        - Linear(out)
    [3. Residual Connection]
    [4. Layer Normalization]
    [5. MLP: Fully Connected Layers]
        - 2개의 FC Layer를 갖는 layer
        - GELU() 사용
            - 입력값과 입력값의 누적 정규 분포의 곱을 사용한 형태
            - 입력값 x가 다른 입력에 비해 얼마나 큰 지에 대한 비율로 값이 조정되기 때문에 확률적인 해석이 가능해지는 장점
    [6. Residual Connection]
    '''
    def __init__(self,
                 dim:         int,    # Embedding Dimension
                 depth:       int,    # Encoder Block의 개수
                 heads:       int,
                 dim_head:    int,
                 mlp_dim:     int
                 ) -> None:
        super().__init__()
        self._norm = nn.LayerNorm(normalized_shape = dim)
        self._layers = nn.ModuleList([])
        for _ in range(depth):
            self._layers.append(nn.ModuleList([
                Attention(dim      = dim,
                          heads    = heads,
                          dim_head = dim_head),
                FeedForwardBlock(dim = dim, hidden_dim = mlp_dim)
            ]))
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x += x  # Resuidal Function
            x = ff(x)
            x += x
        return self.norm(x)
        

class Attention(nn.Module):
    ''' 
    Transformer Scaled Dot-Product-Attention (Attention)
     [1. Q, K, V]
     ---- Input: Q, K ----
     [2. MatMul]
     [3. Scale]
     [4. Mask (option)]
     [5. SoftMax]
     ---- Input: [5], V ----
     [6. MatMul]
    '''
    def __init__(self,
                 dim:       int,        # embedding_dimension
                 heads:     int = 8,    # num_heads
                 dim_head:  int = 64,
                 ) -> None:
        super().__init__()

        self._heads = heads
        self.scale = dim_head ** -0.5        #NOTE: 역수의 제곱근: 1/sqrt(64) --> 0.125 or 1/8
        self.ln = nn.LayerNorm(normalized_shape = dim)
        self.softmax = nn.Softmax(dim = -1)  #NOTE: 입력 텐서의 마지막 차원
                                             # 텐서의 각 행에 대한 클래스 예측 확률 반환
                
        # q or k or v = nn.Linear(dim, dim)
        #TODO: understand
        self._to_qkv = nn.Linear(in_features  = dim, 
                                 out_features = dim * heads * 3, 
                                 bias         = False)
        self._to_out = nn.Linear(in_features  = dim* heads, 
                                 out_features = dim,
                                 bias         = False)
        
    
    def forward(self, x):
        #NOTE: 각 feature에 대해서 정규화 진행 (D 차원에 대해서 정규화)
        x = self.ln(x)

        '''[1. Q, K, V]'''
        # query, key, value
        qkv = self._to_qkv(x).chunk(3, dim = -1)    #NOTE: chunk(input, num_to_chunk, dimension)
        q, k, v = map(lambda t: rearrange(tensor    = t, 
                                          #NOTE 
                                          # b: batch      / n: 나뉘어진 패치 개수
                                          # h: self.heads / d: Embedding dimension
                                          pattern   = 'b n (h d) -> b h n d',
                                          h         = self.heads),
                                          iterables = qkv)
        
        '''[2. MatMul] + [3. Scale]'''
        dots = torch.matmul(input= q, other= k.transpose(-1, -2)) * self.scale

        '''[5. SoftMax]'''
        softmax = self.softmax(dots)
        
        '''[6. MatMul]'''
        out = rearrange(tensor  = out,
                        pattern = 'b h n d -> b n (h d)')   #NOTE: input x와 same shape 출력
        return self._to_out(out)


class FeedForwardBlock(nn.Module):
    '''
    [1. Layer Normalization]
    [2. MLP: Fully Connected Layers]
        - 2개의 FC Layer를 갖는 layer
        - GELU() 사용
    '''
    def __init__(self,
                 dim:           int,   # embedding size
                 hidden_dim:    int,   # dim * expansion
                 drop_p:        float = 0.2
                 ) -> None:
        super().__init__()
        
        self._net = nn.Sequential(
            # [1. Layer Normalization]
            nn.LayerNorm(normalized_shape = dim), 
            
            # [2. MLP: Fully Connected Layers]
            nn.Linear(in_features= dim, out_features= hidden_dim),
            nn.GELU(),
            nn.Linear(in_features= hidden_dim, out_features= dim),
        )

    def forward(self, x):
        return self._net(x)
        


## ViT for CoCa
class SimpleViT(nn.Module):
    """Original ViT
     [1. Patch + Position Embedding (inputs)]
        - PE: Extra learnable [class] embedding
     [2. Linear projection of flattened patches (Embedded Patches)]
     [3. Layer Normalization]
        - torch.nn.LayerNorm()
     [4. Multi-Head Attention: Multi-Headed Self-Attention layer]
        - torch.nn.MultiheadAttention()
     [5. MLP]
        - two torch.nn.Linear() layers
        - torch.nn.GELU()
        - torch.nn.Dropout() layer after each
     [6. Transformer Encoder]
     [7. MLP Head]
    ------------------------------------------------------------
    In SimpleVit, as CoCa Papier mentioned, "CoCa encodes images to latent representations by a neural network encoder, for example, ViT (can be ConvNets as well)"
    we take advantage of Tranformer Encoder as we call this "SimpleViT"
    """
    def __init__(self, *, 
                 image_size:     int, 
                 patch_size:     int, 
                 num_classes:    int, 
                 embedding_dim:  int,     # Size of embedding to turn image into.
                 depth:          int, 
                 heads:          int, 
                 mlp_dim:        int , 
                 channels:       int = 3, 
                 dim_head:       int = 64, 
                 patch_dropout:  float = 0.5
                 ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)  # Tuple (256, 256)
        patch_height, patch_width = pair(patch_size)  # Tuple (32, 32)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        patch_dim = channels * patch_height * patch_width # 3x32x32
        
        '''Preparing pathces of input image'''
        self._to_patch_embedding = nn.Sequential(
            
            ### Patch Embedding ###
            # Normal Input : H x W x C
            # Normal Output: Number of patches, (patch_size**2 * color_channel) 
            # TODO: understand it
            # b, c, (h * patch_h) -> b, h, w, (patch_h * patch_w * channel)
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(normalized_shape= patch_dim), # layer normalization
            nn.Linear(in_features= patch_dim, out_features= embedding_dim),  # 3x32x32 -> dim
            nn.LayerNorm(normalized_shape= embedding_dim)
            )
        
        # self.patch_dropout = PatchDropout(patch_dropout)
        
        self._transformer_encoder = VitTransformerEncoder(dim      = embedding_dim,
                                                          depth    = depth,
                                                          heads    = heads,
                                                          dim_head = dim_head,
                                                          mlp_dim  = mlp_dim)
        
        #NOTE: CoCa encodes images to latent representations 
        #        - differs from the original vit
        self._to_latent = nn.Identity()     #NOTE:  입력과 동일한 tensor를 출력으로 내보내주는 layer
        self._linear_head = nn.Linear(in_features  = embedding_dim, 
                                      out_features = num_classes)
        
    
    def forward(self, img):
        #TODO: *img?
        *_, h, w, dtype = *img.shape, img.dtype
        
        '''[1. Patch + Position Embedding (inputs)]'''
        x = self._to_patch_embedding(img)
        #TODO: understand
        pe = posemb_sincos_2d(x)
        
        #NOTE: ... 
        #       - 다차원 배열에서 특정 축을 생략하는 데 유용
        #       - 예를 들어, x[..., 1]은 "x 배열의 두 번째 열을 선택
        x = rearrange(tensor  =  x,
                      pattern = 'b ... d -> b (...) d') + pe  # (batch, embedding_dim) + pe
        
        # x = self.patch_dropout(x)
        
        x = self._transformer_encoder(x)
        x = x.mean(dim= 1)  # column-wise 연산
        
        #NOTE: CoCa encodes images to latent representations
        x = self._to_latent(x)
        return self._linear_head(x)
   
   
############################################################################################

def identity(t):
    return t

def clone_and_detach(t):
    return t.clone().detach()

def exists(val):
    return val is not None

#TODO
def apply_tuple_or_single(fn : Callable[[], Any], 
                          val,
                            ):
    if isinstance(val, tuple):
        return tuple(map(fn, val))
    return fn(val)

class Extractor(nn.Module):
    """# extractor will enable it so the vision transformer returns its embeddings"""
    def __init__(self,
                 vit:                    torch.nn.Module,
                 device:                 Union[None, str],
                 layer,
                 layer_name:             str = 'transfomer',
                 layer_save_input:       bool = False,
                 return_embeddings_only: bool = False,
                 detach:                 bool = True,
                 ) -> None:
        super().__init__()
        self._vit = vit
        
        self._latents = None
        self._hooks = []
        self._hook_registered = False
        self._device = device
        
        self._layer = layer
        self._layer_name = layer_name
        self._layer_save_input = layer_save_input
        self._return_embeddings_only = return_embeddings_only
        
        self._detach_fn = clone_and_detach if detach else identity
            
    def _hook(self, _, inputs, output):
        #TODO
        layer_output = inputs if self._layer_save_input else output
        self.latents = apply_tuple_or_single(self.detach_fn, layer_output)
    
    def _register_hook(self):
        if not exists(self.layer):
            #TODO : layer???
            assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
            layer = getattr(self.vit, self.layer_name)
        else:
            layer = self.layer
            
        #TODO: layer.register_forward_hook ??
        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self._hook_registered = True
    
    def _clear(self):
        del self._latents
        self._latents = None
    
    def forward(self,
                img,
                return_embeddings_only = False):
        
        self._clear()
        
        if not self._hook_registered:
            self._register_hook()
            
        pred = self.vit(img)
        
        target_device = self.device if exists(self.device) else img.device
        #TODO
        latents = apply_tuple_or_single(fn  = lambda t: t.to(target_device), 
                                        val = self._latents)

        if return_embeddings_only or self._return_embeddings_only:
            return latents
        
        return pred, latents
        