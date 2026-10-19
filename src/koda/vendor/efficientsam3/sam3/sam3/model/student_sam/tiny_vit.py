import torch
import torch.nn as nn
from sam3.backbones.tiny_vit import TinyViT
from .common import LayerNorm2d, UpSampleLayer, OpSequential

__all__ = ['tiny_vit_5m', 'tiny_vit_11m', 'tiny_vit_21m']

class SamTinyViT(TinyViT):
    def __init__(self, img_size=1024, fuse=False, out_indices=None, upsample_mode='bicubic', **kwargs):
        super().__init__(img_size=img_size, **kwargs)
        self.img_size = img_size
        self.fuse = fuse
        self.out_indices = out_indices
        
        embed_dims = kwargs.get('embed_dims')
        if embed_dims is None:
             # infer from depth/heads if possible or assume standard 
             # But standard init passes embed_dims. 
             # If called via wrappers below, embed_dims is passed.
             pass

        if self.fuse:
            # Assuming last two stages are used for fusion
            stage2_channels = embed_dims[2]
            stage3_channels = embed_dims[3]
            self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = embed_dims[-1]

        self.neck = nn.Sequential(
            nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, x):
        # Patch Embed
        x = self.patch_embed(x)
        
        output_dict = dict()
        x = self.layers[0](x)
        output_dict['stem'] = x # Layer 0 is usually stem/conv
        
        # Stages
        # TinyViT layers: 0 is ConvLayer (stem-like), 1, 2, 3 are BasicLayers
        # indices 0, 1, 2, 3 corresponding to stage 0, 1, 2, 3
        
        # self.layers is a ModuleList of 4 layers
        # already called layers[0]
        
        x = self.layers[1](x)
        output_dict['stage1'] = x
        
        x = self.layers[2](x)
        output_dict['stage2'] = x
        
        x = self.layers[3](x)
        output_dict['stage3'] = x
        
        if self.fuse:
            x = self.fuse_stage2(output_dict['stage2']) + self.fuse_stage3(output_dict['stage3'])
        else:
            x = output_dict['stage3']

        x = self.neck(x)
        
        # Return x as features
        return x

def tiny_vit_5m(img_size=1024, **kwargs):
    embed_dims = [64, 128, 160, 320]
    depths = [2, 2, 6, 2]
    num_heads = [2, 4, 5, 10]
    window_sizes = [7, 7, 14, 7]
    drop_path_rate = 0.0
    return SamTinyViT(img_size=img_size, embed_dims=embed_dims, depths=depths, num_heads=num_heads, window_sizes=window_sizes, drop_path_rate=drop_path_rate, **kwargs)

def tiny_vit_11m(img_size=1024, **kwargs):
    embed_dims = [64, 128, 256, 448]
    depths = [2, 2, 6, 2]
    num_heads = [2, 4, 8, 14]
    window_sizes = [7, 7, 14, 7]
    drop_path_rate = 0.1
    return SamTinyViT(img_size=img_size, embed_dims=embed_dims, depths=depths, num_heads=num_heads, window_sizes=window_sizes, drop_path_rate=drop_path_rate, **kwargs)

def tiny_vit_21m(img_size=1024, **kwargs):
    embed_dims = [96, 192, 384, 576]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 18]
    window_sizes = [7, 7, 14, 7]
    drop_path_rate = 0.2
    return SamTinyViT(img_size=img_size, embed_dims=embed_dims, depths=depths, num_heads=num_heads, window_sizes=window_sizes, drop_path_rate=drop_path_rate, **kwargs)

