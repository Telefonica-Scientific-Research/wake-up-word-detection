# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from performer_pytorch import Performer

##### To adapt Audiomer to 2D ######


def make_divisible(v, divisor=8, min_value=None):
    """
    The channel number of each layer should be divisable by 8.
    The function is taken from
    github.com/rwightman/pytorch-image-models/master/timm/models/layers/helpers.py
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        out_channels: int = -1,
        **kwargs: dict,
    ):
        super().__init__()
        assert in_channels > 0

        num_reduced_channels = make_divisible(
            max(out_channels, 8) // reduction, 8
        )

        self.fc1 = nn.Conv2d(in_channels, num_reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(num_reduced_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = F.adaptive_avg_pool2d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x).sigmoid()
        return x


def to_channels_last_2d(x):
    return rearrange(x, "b channels f_bins t_bins -> b t_bins (f_bins channels)")


def to_frames_last_2d(x, f_bins, channels):
    return rearrange(x, "b t_bins (f_bins channels) -> b channels f_bins t_bins", f_bins=f_bins, channels=channels)


class SepConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False
    ):
        assert(stride[1] < kernel_size[1])
        super().__init__()
        padding = kernel_size[1] - stride[1] # + 1
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, padding),
            groups=in_channels,
            bias=bias
            )
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            padding=(0, 1)
            )

    def forward(self, x):
        # x.shape -> (b, channels, f_bins, t_bins)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class MBConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        expansion_factor=2,
        use_se=True
    ):
        super().__init__()
        if stride[1] > kernel_size[1]:
            print(stride[1])
            print(kernel_size[1])

        assert(stride[1] > 1)
        self.use_se = use_se
        self.sep_conv = SepConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, bias=bias
            )
        if self.use_se:
            self.se = SqueezeExcitation2D(
                in_channels=out_channels
                )

    def forward(self, inp):
        x = self.sep_conv(inp)
        if self.use_se:
            x = x * self.se(x)
        return x


class ConvEmbedding2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, expansion_factor, use_se, equal_strides):
        super().__init__()
        context_stride = stride // 2 if equal_strides else stride
        self.to_q = MBConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride // 2),
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )
        self.to_context = MBConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, context_stride),
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )

    def forward(self, x):
        q = self.to_q(x)
        context = self.to_context(x)
        return q, context


class AudiomerEncoderBlock2D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        dim_head=64, 
        depth=1, 
        num_heads=6, 
        expansion_factor=2,
        use_attention=True,
        use_se=True,
        equal_strides=False
        ):

        super().__init__()
        stride = kernel_size - 1
        self.use_attention = use_attention

        self.conv = ConvEmbedding2D(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            use_se=use_se,
            equal_strides=equal_strides
        )
        
        if self.use_attention:
            self.performer = Performer(
                dim=out_channels*5,
                depth=depth,
                heads=num_heads,
                dim_head=dim_head,
                ff_glu=True,
                attn_dropout=0.2,
                use_scalenorm=True,
                ff_mult=expansion_factor,
                generalized_attention = True,
                kernel_fn=nn.ReLU(),
            )

    def forward(self, x):
        # (b, in_channels, f_bins, t_frames) -> (b, t_frames, (f_bins*out_channels))
        q, context = self.conv(x)
        _, channels, f_bins, _ = q.shape
        q = to_channels_last_2d(q)
        context = to_channels_last_2d(context)
        
        if self.use_attention:
            out = q + self.performer(
                q,
                context=context,
                context_mask=torch.ones_like(context).bool()
                )
        else:
            out = q
                   
        # (b, t_frames, (f_bins*out_channels) -> (b, out_channels, f_bins, t_frames)
        out = to_frames_last_2d(out, f_bins=f_bins, channels=channels)
        return out


class AudiomerEncoder2D(nn.Module):
    def __init__(self, config, kernel_sizes, dim_head, depth, num_heads, use_residual, use_cls, expansion_factor, input_size, use_se, use_attention, equal_strides, **kwargs):
        super().__init__()
        assert(len(kernel_sizes) == len(config) - 1)

        if use_cls:
            input_size += 128
        
        self.layers = []
        self.identity_layers = []
        self.use_residual = use_residual
        self.use_cls = use_cls
        paddings = [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        #paddings = [4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]

        for i in range(len(config) - 1):
            in_channels, out_channels = config[i], config[i+1]
            padding = paddings[i]
            print(i)
            self.layers.append(
                AudiomerEncoderBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    num_heads=num_heads,
                    depth=depth,
                    dim_head=dim_head if isinstance(
                        dim_head, int) else dim_head[i],
                    expansion_factor=expansion_factor,
                    use_se=use_se, 
                    use_attention=use_attention, 
                    equal_strides=equal_strides
                )
            )
            
            if use_residual:
                self.identity_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=(1, kernel_sizes[i] // 2),
                            stride=(1, kernel_sizes[i] // 2),
                            bias=False,
                            padding=(0, padding)
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            input_size = math.floor(
                (input_size + kernel_sizes[i] // 2 + 1) / (kernel_sizes[i] // 2)) + 1

        self.layers = nn.ModuleList(self.layers)
        if use_residual:
            self.identity_layers = nn.ModuleList(self.identity_layers)
        else:
            self.identity_layers = [None] * len(self.layers)

    def forward(self, x):

        for (layer, id_layer) in zip(self.layers, self.identity_layers):
            x_copy = x

            # Convolutional Attention
            x = layer(x)
            
            # Residual path
            if self.use_residual:
                x_copy = id_layer(x_copy)
                x = x + x_copy
        return x
