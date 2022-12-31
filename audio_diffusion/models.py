import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

from .blocks import SkipBlock, FourierFeatures, SelfAttention1d, ResConvBlock, Downsample1d, Upsample1d
from .utils import append_dims, expand_to_planes
from .autoencoders import AudioAutoencoder

class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self, 
        global_args, 
        io_channels = 2, 
        depth=14, 
        n_attn_layers = 6,
        c_mults = [128, 128, 256, 256] + [512] * 10
    ):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = ResConvBlock

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d(kernel="cubic")
                    # nn.Upsample(scale_factor=2, mode='linear',
                    #             align_corners=False),
                )
            else:
                block = nn.Sequential(
                    conv_block(io_channels + 16 + global_args.latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond=None):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        
        inputs = [input, timestep_embed]

        if cond is not None:
            cond = F.interpolate(cond, (input.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        return self.net(torch.cat(inputs, dim=1))


class DiffusionUnet1D(nn.Module):
    def __init__(
        self, 
        io_channels = 2, 
        depth=14,
        n_attn_layers = 6,
        channels = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        learned_resample = False,
        strides = [2] * 14
    ):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = partial(ResConvBlock)

        for i in range(depth, 0, -1):
            c = channels[i - 1]
            stride = strides[i-1]
            if i > 1:
                c_prev = channels[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if learned_resample else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                    # nn.Upsample(scale_factor=2, mode='linear',
                    #             align_corners=False),
                )
            else:
                block = nn.Sequential(
                    conv_block((io_channels + cond_dim) + 16, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t, cond=None):

        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        inputs = [x, timestep_embed]

        if cond is not None:
            cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        return outputs

class LatentAudioDiffusion(nn.Module):
    def __init__(
        self, 
        autoencoder: AudioAutoencoder,
        **model_kwargs
    ):
        super().__init__()

        default_model_kwargs = {"io_channels": 32, "n_attn_layers": 4, "channels": [512]*6 + [1024]*4, "depth": 10}

        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        self.diffusion = DiffusionUnet1D(**{
            **default_model_kwargs,
            **model_kwargs
        })

        self.autoencoder = autoencoder

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents):
        return self.autoencoder.decode(latents)

    def forward(self, x, t, **extra_args):
        return self.diffusion(x, t, **extra_args)