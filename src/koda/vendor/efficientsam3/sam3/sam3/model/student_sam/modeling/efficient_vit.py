import torch
import torch.nn as nn
from sam3.backbones.efficientvit.efficientvit.backbone import (
    EfficientViTBackbone,
    efficientvit_backbone_b0 as _efficientvit_backbone_b0,
    efficientvit_backbone_b1 as _efficientvit_backbone_b1,
    efficientvit_backbone_b2 as _efficientvit_backbone_b2
)
from .common import LayerNorm2d, UpSampleLayer, OpSequential

__all__ = ['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2']

class SamEfficientViT(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone, fuse=False, upsample_mode='bicubic'):
        super().__init__()
        self.backbone = backbone
        self.fuse = fuse
        self.img_size = 1024 # Default assumption

        width_list = backbone.width_list
        # width_list has [stem, stage1, stage2, stage3, stage4]
        # e.g. b0: [8, 16, 32, 64, 128]
        
        if self.fuse:
            stage3_channels = width_list[-2]
            stage4_channels = width_list[-1]
            self.fuse_stage3 = nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage4 = OpSequential([
                nn.Conv2d(stage4_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = width_list[-1]

        self.neck = nn.Sequential(
            nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, x):
        out = self.backbone(x)
        # output_dict = {"input": x, "stage0": ..., "stage1": ..., "stage_final": ...}
        # backbone.forward loop:
        # stage_id starts at 1.
        # output_dict["stage%d" % stage_id]
        
        # We need last two stages if fusing.
        # width_list has len 5. Stages are 4.
        # stage4 is final.
        # stage3 is previous.
        
        if self.fuse:
            s3 = out['stage3']
            s4 = out['stage4'] # or 'stage_final' which is same as last stage output
            x = self.fuse_stage3(s3) + self.fuse_stage4(s4)
        else:
            x = out['stage_final']

        x = self.neck(x)
        return x

def efficientvit_b0(img_size=1024, **kwargs):
    # kwargs might contain fuse, upsample_mode
    fuse = kwargs.pop('fuse', False)
    upsample_mode = kwargs.pop('upsample_mode', 'bicubic')
    backbone = _efficientvit_backbone_b0(**kwargs)
    return SamEfficientViT(backbone, fuse=fuse, upsample_mode=upsample_mode)

def efficientvit_b1(img_size=1024, **kwargs):
    fuse = kwargs.pop('fuse', False)
    upsample_mode = kwargs.pop('upsample_mode', 'bicubic')
    backbone = _efficientvit_backbone_b1(**kwargs)
    return SamEfficientViT(backbone, fuse=fuse, upsample_mode=upsample_mode)

def efficientvit_b2(img_size=1024, **kwargs):
    fuse = kwargs.pop('fuse', False)
    upsample_mode = kwargs.pop('upsample_mode', 'bicubic')
    backbone = _efficientvit_backbone_b2(**kwargs)
    return SamEfficientViT(backbone, fuse=fuse, upsample_mode=upsample_mode)

