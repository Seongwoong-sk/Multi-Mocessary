# CoCa
# consists of image encodoer, unimodal text decoder, multimodal text decoder
# 1. CoCa encodes images to latent representations by a neural network encoder, for example, ViT (can be ConvNets as well)
# (4p) - decodes texts with a casual masking transformer decoder
# (5p) - unlike standard decoder transformers, Coca omits cross-attention in the first half of the decoder layers to encode unimodal text respresentations 
#      - and cascades the rest of the decoder layers, cross-attending to the image encoder for multimodal image-text representations

# image encoder --> ViT based encoder
# unimodal text decoder --> language-model transformers
# multimodal text decoder --> 

# https://blog.research.google/2022/05/image-text-pre-training-with.html?m=1


import torch
import numpy as np
from CoCa.models.encoder import SimpleViT, Extractor              


vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    embedding_dim = 1024,
    depth = 6, 
    heads = 16,
    mlp_dim = 2048,
    patch_dropout = 0.5  # https://arxiv.org/abs/2212.00794
)

vit = Extractor(vit, return_embeddings_only = True, detach = False)

print(vit)