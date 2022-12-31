# Modified from https://github.com/wesbz/SoundStream/blob/main/net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

# Generator
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
        )

    def forward(self, x):
        return self.layers(x)


class AudioEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=64, latent_dim=328, c_mults = [2, 4, 8, 16, 32], strides = [2, 2, 2, 2, 2, 2]):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1):
            layers.append(EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=64, latent_dim=32, c_mults = [2, 4, 8, 16, 32], strides = [2, 2, 2, 2, 2, 2]):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers.append(DecoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i-1]*channels, stride=strides[i-1]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        channels = 64,
        c_mults = [2, 4, 8, 16, 32],        
        strides = [2, 2, 2, 2, 2],
        latent_dim = 32,
        in_channels = 2,
        out_channels = 2
    ):
        super().__init__()

        self.downsampling_ratio = np.prod(strides)

        self.latent_dim = latent_dim

        self.encoder = AudioEncoder(
            in_channels = in_channels, 
            channels = channels, 
            latent_dim = self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = AudioDecoder(
            out_channels=out_channels, 
            channels = channels, 
            latent_dim = self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

    def encode(self, audio,):
        return torch.tanh(self.encoder(audio))

    def decode(self, latents):
        return self.decoder(latents)