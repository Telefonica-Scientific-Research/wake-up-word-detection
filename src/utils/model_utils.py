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

from einops import rearrange, repeat

from performer_pytorch import Performer


## Serializable Model ##

class SerializableModule(nn.Module):

    subclasses = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, arch, input_shape, num_classes):
        if arch not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arch))

        return cls.subclasses[arch](input_shape, num_classes)

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.jit')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


## Audiomer ##

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


class SqueezeExcitation(nn.Module):
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

        self.fc1 = nn.Conv1d(in_channels, num_reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(num_reduced_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = F.adaptive_avg_pool1d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x).sigmoid()
        return x


class SepConv1d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False
    ):
        assert(stride < kernel_size)
        super().__init__()
        padding = kernel_size - stride # + 1
        self.depthwise = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
            )
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.pointwise = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            padding=1
            )

    def forward(self, x):
        # x.shape -> (b, channels, frames)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class MBConv(nn.Module):
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
        assert(stride > 1)
        self.use_se = use_se
        self.sep_conv = SepConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, bias=bias
            )
        if self.use_se:
            self.se = SqueezeExcitation(
                in_channels=out_channels
                )

    def forward(self, inp):
        x = self.sep_conv(inp)
        if self.use_se:
            x = x * self.se(x)
        return x

####### separation

def to_channels_last(x):
    return rearrange(x, "b channels frames -> b frames channels")


def to_frames_last(x):
    return rearrange(x, "b frames channels -> b channels frames")


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, expansion_factor, use_se, equal_strides):
        super().__init__()
        context_stride = stride // 2 if equal_strides else stride
        self.to_q = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride // 2,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )
        self.to_context = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=context_stride,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )

    def forward(self, x):
        q = self.to_q(x)
        q = to_channels_last(q)

        context = self.to_context(x)
        context = to_channels_last(context)
        return q, context


class AudiomerEncoderBlock(nn.Module):
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

        self.conv = ConvEmbedding(
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
                dim=out_channels,
                depth=depth,
                heads=num_heads,
                dim_head=dim_head,
                ff_glu=True,
                attn_dropout=0.2,
                use_scalenorm=True,
                ff_mult=expansion_factor,
                feature_redraw_interval=1000, # Gaussian aproximation
                generalized_attention = True,
                kernel_fn=nn.ReLU(),
            )

    def forward(self, x):
        # (b, in_channels, input_frames) -> (b, num_frames, out_channels)
        q, context = self.conv(x)
        # (b, num_frames, out_channels) -> (b, num_frames, out_channels)
        if self.use_attention:
            out = q + self.performer(
                q,
                context=context,
                context_mask=torch.ones_like(context).bool()
                )
        else:
            out = q            
        # (b, num_frames, out_channels) -> (b, out_channels, num_frames)
        out = to_frames_last(out)
        return out


class AudiomerEncoder(nn.Module):
    def __init__(self, config, kernel_sizes, dim_head, depth, num_heads, use_residual, use_cls, expansion_factor, input_size, use_se, use_attention, equal_strides, **kwargs):
        super().__init__()
        assert(len(kernel_sizes) == len(config) - 1)
        if use_cls:
            input_size += 128
        self.layers = []
        self.identity_layers = []
        self.use_residual = use_residual
        self.use_cls = use_cls
        #paddings = [3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4] # cls
        paddings = [3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4] # max, mean
        for i in range(len(config) - 1):
            in_channels, out_channels = config[i], config[i+1]
            padding = paddings[i]
            self.layers.append(
                AudiomerEncoderBlock(
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
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_sizes[i] // 2,
                            stride=kernel_sizes[i] // 2,
                            bias=False,
                            padding=padding
                        ),
                        nn.BatchNorm1d(out_channels),
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
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(
                1, 1, 128))

    def forward(self, x):
        if self.use_cls:
            cls_tokens = repeat(
                self.cls_token, '() n d -> b n d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=2)

        for (layer, id_layer) in zip(self.layers, self.identity_layers):
            x_copy = x
            x = layer(x)
            if self.use_residual:
                x_copy = id_layer(x_copy)
                x = x + x_copy
        return x


### Lambda layer ####

class LambdaLayer1D(nn.Module):
    def __init__(self, d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='conv', dim=None):
        super().__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, d_out // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2
        self.layer_type = layer_type

        if layer_type == 'conv':
            self.queries = nn.Sequential(
                nn.Conv1d(d_in, k * heads, kernel_size=1, bias=False),
                nn.BatchNorm1d(k * heads)
            )
            self.keys = nn.Sequential(
                nn.Conv1d(d_in, k * u, kernel_size=1, bias=False),
            )
            self.values = nn.Sequential(
                nn.Conv1d(d_in, self.vv * u, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.vv * u)
            )
        if layer_type == 'dense':
            self.queries = nn.Sequential(
                nn.Linear(d_in, k*heads, bias=True),
                nn.BatchNorm1d(dim)
            )
            self.keys = nn.Sequential(
                nn.Linear(d_in, k * u, bias=True),
            )
            self.values = nn.Sequential(
                nn.Linear(d_in, self.vv * u, bias=True),
                nn.BatchNorm1d(dim)
            )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, d, m = x.size()

        # Permute the variable to compute the q,k,v with a dense operator
        if self.layer_type == 'dense':
            x = x.permute(0,2,1)
        
        queries = self.queries(x).view(n_batch, self.heads, self.kk, m) # b, heads, k // heads, m
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, m)) # b, k, u, m
        values = self.values(x).view(n_batch, self.vv, self.uu, m) # b, v, u, m

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, m)
            lambda_p = F.conv2d(values, self.embedding, padding=(0, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, m)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, m)

        return out

###### BC-Resnet-1 ######

# https://arxiv.org/abs/2103.13620
class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels*sub_bands, eps=eps)

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.sub_bands, F // self.sub_bands, T)
        x = self.bn(x)
        return x.view(N, C, F, T)


class NormalBlock(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = 0.1, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = 0.1, use_subspectral: bool = True):
        super().__init__()

        if stride == 1:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same")
        else:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan, padding=(1, 0))

        norm_layer = SubSpectralNorm(out_chan, 5) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)