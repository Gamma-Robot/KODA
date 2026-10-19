import torch
import torch.nn as nn
from .common import LayerNorm2d, UpSampleLayer, OpSequential
from sam3.backbones.repvit import RepViTBlock, Conv2d_BN, _make_divisible

__all__ = ['rep_vit_m1', 'rep_vit_m2', 'rep_vit_m3', 'RepViT']

m1_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 48, 1, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 96, 0, 0, 2],
    [3, 2, 96, 1, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 192, 0, 1, 2],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 384, 0, 1, 2],
    [3, 2, 384, 1, 1, 1],
    [3, 2, 384, 0, 1, 1]
]

m2_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]

m3_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]


class RepViT(nn.Module):
    arch_settings = {
        'm1': m1_cfgs,
        'm2': m2_cfgs,
        'm3': m3_cfgs
    }

    def __init__(self, arch, img_size=1024, fuse=False, freeze=False,
                 load_from=None, use_rpn=False, out_indices=None, upsample_mode='bicubic'):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.fuse = fuse
        self.freeze = freeze
        self.use_rpn = use_rpn
        self.out_indices = out_indices

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            
            if not self.fuse and c in [384, 512]:
                s = 1
            
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)

        if self.fuse:
            stage2_channels = _make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
            stage3_channels = _make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
            self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = output_channel
        self.neck = nn.Sequential(
            nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        if load_from is not None:
            state_dict = torch.load(load_from)['model']
            new_state_dict = dict()
            use_new_dict = False
            for key in state_dict:
                if key.startswith('image_encoder.'):
                    use_new_dict = True
                    new_key = key[len('image_encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
            if use_new_dict:
                state_dict = new_state_dict
            print(self.load_state_dict(state_dict, strict=True))

    def train(self, mode=True):
        super(RepViT, self).train(mode)
        if self.freeze:
            self.features.eval()
            self.neck.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        output_dict['stem'] = x
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1

        if self.fuse:
            x = self.fuse_stage2(output_dict['stage2']) + self.fuse_stage3(output_dict['stage3'])

        x = self.neck(x)
        output_dict['final'] = x

        if self.use_rpn:
            if self.out_indices is None:
                self.out_indices = [len(output_dict) - 1]
            out = []
            for i, key in enumerate(output_dict):
                if i in self.out_indices:
                    out.append(output_dict[key])
            return tuple(out)
        return x


def rep_vit_m1(img_size=1024, **kwargs):
    return RepViT('m1', img_size, **kwargs)


def rep_vit_m2(img_size=1024, **kwargs):
    return RepViT('m2', img_size, **kwargs)


def rep_vit_m3(img_size=1024, **kwargs):
    return RepViT('m3', img_size, **kwargs)
