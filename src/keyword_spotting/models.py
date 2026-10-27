# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import torch
import torch.nn as nn
import torchaudio

from typing import List

from conformer import Conformer
from performer_pytorch import Performer

from utils.model_utils import *
from utils.audiomer2d_utils import *
from utils.lambda_utils import *

import utils.sonopytorch as sonopytorch


def get_conv2D_output_shape(input_shape, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """ 
    Input: (Hin,Win)
    Output: (Hout,Wout)
    """
    h_out = int((input_shape[0] + 2*padding[0] - dilation[0]*(kernel[0] - 1) - 1)/stride[0] + 1)
    w_out = int((input_shape[1] + 2*padding[1] - dilation[1]*(kernel[1] - 1) - 1)/stride[1] + 1)
    return (h_out, w_out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x, pool_size : List[int] = (2, 2)):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.avg_pool2d(x, pool_size)
        return x


class TwoInputSequential(nn.Sequential):
    def forward(self, x, y):
        for module in self._modules.values():
            x,y = module(x, y)
        return x, y


class SpeechResnetX(SerializableModule):
    """
    ResNetX architecture for KWS.
    """
    def __init__(self, num_classes, n_layers, n_maps, res_pool=None, dilation=True):
        super().__init__()

        self.n_layers = n_layers

        # Fisrt convolution
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if(res_pool):
            self.pool = nn.AvgPool2d(res_pool)

        # Create convolutional layers
        if(dilation):
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        
        # Add Convolutional layers with BatchNormalization
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        
        # Classifier
        self.fc = nn.Linear(n_maps, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        for i in range(self.n_layers + 1):
            y = nn.functional.relu(getattr(self, "conv{}".format(i))(x))
            # If average pool after first layer
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.fc(x)
    
    def get_embedding(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = nn.functional.relu(getattr(self, "conv{}".format(i))(x))
            # If average pool after first layer
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        return x


@SerializableModule.register_model('resnet15')
class Resnet15(SerializableModule):
    """
    ResNet15 architecture for KWS detection.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        self.model = SpeechResnetX(num_classes, n_layers=13, n_maps=45, res_pool=None, dilation=True)

    def forward(self, x):
        x.transpose_(2, 1) # coherence
        return self.model(x)
    
    def get_embedding(self, x):
        x.transpose_(2, 1) # coherence
        return self.model.get_embedding(x)


@SerializableModule.register_model('resnet15-narrow')
class Resnet15Narrow(SerializableModule):
    """
    ResNet15 narrow architecture for KWS detection.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        self.model = SpeechResnetX(num_classes, n_layers=13, n_maps=19, res_pool=None, dilation=True)

    def forward(self, x):
        x.transpose_(2, 1) # coherence
        return self.model(x)
    
    def get_embedding(self, x):
        x.transpose_(2, 1) # coherence
        return self.model.get_embedding(x)


@SerializableModule.register_model('resnet8')
class Resnet8(SerializableModule):
    """
    ResNet8 architecture for KWS detection.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        self.model = SpeechResnetX(num_classes, n_layers=6, n_maps=45, res_pool=(4, 3), dilation=False)

    def forward(self, x):
        x.transpose_(2, 1) # coherence
        return self.model(x)


@SerializableModule.register_model('sgru')
class SGRU(SerializableModule):
    """
    RNN-based KWS model.
    Inheritance form Mycroft Precise engine.
    """
    def __init__(self, input_shape, num_classes=2, input_dim=13, hidden_dim=200, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_shape[1], self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear( self.hidden_dim, num_classes)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:,-1])
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
    
    def get_embedding(self, x):
        h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        x, _ = self.gru(x, h)
        return x
    

@SerializableModule.register_model('sgru_max')
class SGRU_MAX(SerializableModule):
    """
    RNN-based KWS model.
    Max pooling in the hidden states variations over time.
    """
    def __init__(self, input_shape, num_classes=2, input_dim=13, hidden_dim=200, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = input_shape[1]
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.pooling = nn.MaxPool2d(kernel_size=(input_shape[0], 1))
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.pooling(out).squeeze(1))
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


@SerializableModule.register_model('sgru_avg')
class SGRU_AVG(SerializableModule):
    """
    RNN-based KWS model.
    Avergae Pooling in the hiddents states variations over time.
    """
    def __init__(self, input_shape, num_classes=2, input_dim=13, hidden_dim=200, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = input_shape[1]
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.pooling = nn.AvgPool2d(kernel_size=(input_shape[0], 1))
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.pooling(out).squeeze(1))
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


@SerializableModule.register_model('sgru_att')
class GRU_ATT(SerializableModule):
    """
    RNN-based KWS model.
    Avergae Pooling in the hiddents states variations over time.
    """
    def __init__(self, input_shape, num_classes=2, input_dim=13, hidden_dim=200, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = input_shape[1]
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.linear_attention = Performer(
                dim=hidden_dim,
                depth=1,
                heads=5,
                dim_head=64,
                ff_glu=True,
                attn_dropout=0.2,
                use_scalenorm=True,
                ff_mult=2,
                generalized_attention = True,
                kernel_fn=nn.ReLU(),
            )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x, h):
        x, h = self.gru(x, h)
        cls_tokens = repeat(self.cls_token, '() t h -> b t h', b=x.shape[0]) # CLS
        x = torch.cat((cls_tokens, x), dim=-2) # CLS
        x = self.linear_attention(x)
        x = x[:, 0, :] # CLS
        out = self.fc(x)
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


@SerializableModule.register_model('cnn_trad_pool2')
class CNN_TRAD_POOL2(SerializableModule):
    """
    Based on Honk implementation of KWS model.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        features_shape = self.__get_features_shape(input_shape[::-1])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(20, 8), stride=(1, 1))
        self.truncated_normal(self.conv1.weight.data)
        self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4), stride=(1, 1))
        self.truncated_normal(self.conv2.weight.data)
        self.conv2.bias.data.zero_()
        # FIXME: this layer does not do anything
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
        self.lin = nn.Linear(in_features=features_shape[0]*features_shape[1]*64, out_features=num_classes, bias=True)
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        
    def forward(self, x):
        x.transpose_(2, 1) # coherence
        x = x.unsqueeze(1) # add channel dimension

        # First convolutional and pool
        x = nn.functional.relu(self.conv1(x)) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)

        # Second convolutional and pool
        x = nn.functional.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
        x = self.dropout(x)
        x = self.pool2(x)
        
        # Classifier
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        x = self.lin(x)

        return x
    
    def truncated_normal(self, tensor, std_dev=0.01):
        tensor.zero_()
        tensor.normal_(std=std_dev)
        while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
            t = tensor[torch.abs(tensor) > 2 * std_dev]
            t.zero_()
            tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

    def __get_features_shape(self, input_shape):
        # Conv1
        output_shape = get_conv2D_output_shape(input_shape, kernel=(20, 8))
        # Pooling  1
        output_shape = get_conv2D_output_shape(output_shape, kernel=(2, 2), stride=(2, 2))
        # Conv2
        output_shape = get_conv2D_output_shape(output_shape, kernel=(10, 4))
        return output_shape


@SerializableModule.register_model('cnn_trad_pool2s')
class CNN_TRAD_POOL_2S(SerializableModule):
    """
    Based on Honk implementation of KWS model.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        features_shape = self.__get_features_shape(input_shape[::-1])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(6, 8), stride=(1, 1))
        self.truncated_normal(self.conv1.weight.data)
        self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 4), stride=(1, 1))
        self.truncated_normal(self.conv2.weight.data)
        self.conv2.bias.data.zero_()
        # FIXME: this layer does not do anything
        #self.pool2 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1), padding=0, dilation=1, ceil_mode=False)
        self.lin = nn.Linear(in_features=features_shape[0]*features_shape[1]*64, out_features=num_classes, bias=True)
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        
    def forward(self, x):
        x.transpose_(2, 1)
        x = x.unsqueeze(1) # add channel dimension

        # First convolutional and pool
        x = nn.functional.relu(self.conv1(x)) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)

        # Second convolutional and pool
        x = nn.functional.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
        x = self.dropout(x)
        #x = self.pool2(x)
        
        # Classifier
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        x = self.lin(x)

        return x
    
    def truncated_normal(self, tensor, std_dev=0.01):
        tensor.zero_()
        tensor.normal_(std=std_dev)
        while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
            t = tensor[torch.abs(tensor) > 2 * std_dev]
            t.zero_()
            tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

    def __get_features_shape(self, input_shape):
        # Conv1
        output_shape = get_conv2D_output_shape(input_shape, kernel=(6, 8))
        # Pooling  1
        output_shape = get_conv2D_output_shape(output_shape, kernel=(2, 2), stride=(2, 2))
        # Conv2
        output_shape = get_conv2D_output_shape(output_shape, kernel=(3, 4))
        return output_shape


@SerializableModule.register_model('cnn_fat2019')
class CNN_2D(SerializableModule):
    """
    Adapted fromn Kaggle's FAT 2019 competition.
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x.unsqueeze_(1)
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x
    
    def get_embedding(self, x):
        x.unsqueeze_(1)
        x = self.conv(x)
        # Average pooling in the frequency dimension
        # x = torch.mean(x, dim=3)
        return x


@SerializableModule.register_model('lenet')
class LeNet(SerializableModule):
    """
    Baseline classifier.
    """
    def __init__(self, input_shape, num_classes=2):
        super(LeNet, self).__init__()

        features_shape = self.__get_features_shape((input_shape[1], input_shape[0]))
        
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1),
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
    
        self.classifier = nn.Sequential(     
            nn.Linear(20*features_shape[0]*features_shape[1], 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x.transpose_(2, 1)
        x.unsqueeze_(1) # Add channel dimension
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
    def __get_features_shape(self, input_shape):
        # Conv1
        output_shape = get_conv2D_output_shape(input_shape, kernel=(5, 5))
        # Max Pooling  1
        output_shape = get_conv2D_output_shape(output_shape, kernel=(2, 2), stride=(2, 2))
        # Conv2
        output_shape = get_conv2D_output_shape(output_shape, kernel=(5, 5))
        # Max Pooling  2
        output_shape = get_conv2D_output_shape(output_shape, kernel=(2, 2), stride=(2, 2))
        return output_shape


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    

@SerializableModule.register_model('cnn10')
class CNN10(SerializableModule):
    """
    As implemented in: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
    """
    def __init__(self, input_shape, num_classes=10):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(input_shape[1])# [b, f_bins, t_bins]

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, num_classes, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
 
    def forward(self, x):
        """
        Input: (batch_size, data_length)
        """
        x = x.unsqueeze(1) # add channel dimension
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu_(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


# Audiomer2D
@SerializableModule.register_model('audiomer-2d')
class Audiomer2D(SerializableModule):
    def __init__(self, input_shape, num_classes):
        '''config[-1] will be the output sequence length'''
        super().__init__()

        config = [4, 16, 32, 64]
        num_heads = 2
        dim_head = 32
        depth = 2
        mlp_dim = config[-1]
        mlp_dropout = 0.2
        kernel_sizes = [5] * (len(config)-1)
        pool = 'cls'
        use_residual = True
        use_attention = True
        equal_strides = False 
        use_se = True # Improves marginally and saves computational load
        # Performer
        expansion_factor = 2
        input_size = 151

        self.pool = pool
        self.use_cls = True if self.pool == "cls" else False

        # Initial frequency reduction
        self.f_reduction = torch.nn.Conv2d(1, 4, kernel_size=[15, 1], stride=[6, 1])

        self.encoder = AudiomerEncoder2D(
            config=config,
            kernel_sizes=kernel_sizes,
            num_heads=num_heads,
            depth=depth,
            use_residual=use_residual,
            use_cls=self.use_cls,
            dim_head=dim_head,
            expansion_factor=expansion_factor,
            input_size=input_size,
            use_attention=use_attention,
            use_se=use_se,
            equal_strides=equal_strides
            )

        self.classifier = nn.Sequential(
            nn.Linear(config[-1], mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(
                1, 1, 40, 2))

    def forward(self, x):
        x.transpose_(2, 1)
        
        # Add channel dim
        x = x.unsqueeze(1)

        # CLS: add random token
        if self.use_cls:
            cls_tokens = repeat(
                self.cls_token, '() n f t -> b n f t', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=3)

        # Frequency dimendion reduction
        x = self.f_reduction(x)

        # Audiomer Encoder
        x = self.encoder(x)
        
        # Pooling
        if self.pool == "mean":
            x = torch.mean(x, dim=3)
            x = torch.mean(x, dim=2)
        elif self.pool == "max":
            x, _ = torch.max(x, dim=3)
            x, _ = torch.max(x, dim=2)
        else:
            # CLS
            x = x[:, :, :, 0] 
            x, _ = torch.max(x, dim=2)
        
        # Classification
        x = self.classifier(x)
        return x


# Audiomer-1D
class AudiomerClassification(SerializableModule):
    def __init__(self, config, input_shape, num_classes):
        '''config[-1] will be the output sequence length'''
        super().__init__()

        num_heads = 2
        dim_head = 8
        depth = 1
        mlp_dim = config[-1]
        mlp_dropout = 0.2
        kernel_sizes = [5] * (len(config)-1)
        pool = 'mean'
        use_residual = True
        use_attention = True
        use_se = True
        equal_strides = False # improves marginally and saves computational load
        expansion_factor = 2
        input_size = 8192
        #self.resampler = torchaudio.transforms.Resample(16000, 8192).cuda() # remove
        self.resampler = torchaudio.transforms.Resample(16000, 8192).cpu() # remove

        self.pool = pool
        self.use_cls = True if self.pool == "cls" else False

        self.encoder = AudiomerEncoder(
            config=config,
            kernel_sizes=kernel_sizes,
            num_heads=num_heads,
            depth=depth,
            use_residual=use_residual,
            use_cls=self.use_cls,
            dim_head=dim_head,
            expansion_factor=expansion_factor,
            input_size=input_size,
            use_attention=use_attention,
            use_se=use_se,
            equal_strides=equal_strides
            )

        self.classifier = nn.Sequential(
            nn.Linear(config[-1], mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resampler(x)
        x = self.encoder(x)
        if self.pool == "mean":
            x = x.mean(dim=2)
        elif self.pool == "max":
            x, _ = torch.max(x, dim=2)
        else:
            x = x[:, :, 0] # CLS
        x = self.classifier(x)
        return x


@SerializableModule.register_model('audiomer-l')
class AudiomerL(SerializableModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Hardcode Audiomer-L
        config = [1, 4, 8, 16, 16, 32, 32, 64, 64, 96, 96, 192]
        self.audiomer_classifier = AudiomerClassification(config, input_shape, num_classes)

    def forward(self, x):
        return self.audiomer_classifier(x)


@SerializableModule.register_model('audiomer-s')
class AudiomerS(SerializableModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Hardcode Audiomer-S
        config = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64]
        self.audiomer_classifier = AudiomerClassification(config, input_shape, num_classes)

    def forward(self, x):
        return self.audiomer_classifier(x)


@SerializableModule.register_model('bcresnet')
class BcResNetModel(SerializableModule):
    """
    BCResNet-1 as defined in https://arxiv.org/pdf/2106.04140v2.pdf
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()

        scale = 1
        dropout = 0.1
        use_subspectral = True

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(8*scale, 12*scale, dilation=2, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock(12*scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock(12*scale, 16*scale, dilation=4, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(16*scale, 20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        x.transpose_(2, 1)
        x = x.unsqueeze(1)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        # Pooling
        #x = torch.mean(x, dim=3, keepdim=True)  # mean in time dim
        #x, _ = torch.max(x, dim=3, keepdim=True)  # max in time dim
        
        x = self.head_conv(x)
        
        # Attention: input dim [b, classes, 1, t]
        w = torch.nn.functional.softmax(x, dim=1) # softmax in class dim
        x = (x*w).sum(dim=-1) / w.sum(dim=-1)

        x = x.squeeze(2)
        return x
    
    def get_embedding(self, x: torch.Tensor):
        x.transpose_(2, 1)
        x = x.unsqueeze(1)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)
        
        x = self.head_conv(x) # [b, classes, 1, t]
        
        return x     
    

@SerializableModule.register_model('conformer')
class ConformerClassifier(SerializableModule):
    """
    Conformer-based model.
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        self.confomer = Conformer(
            dim = input_shape[0],
            depth = 12, # 12 blocks
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            conv_expansion_factor = 2,
            conv_kernel_size = 31,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            conv_dropout = 0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_shape[1], 100),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x.transpose_(2, 1)
        x = self.confomer(x)
        x = x[:, :, 0] # CLS
        x = self.classifier(x)
        return x
    

@SerializableModule.register_model('lambda-resnet18')
class LambdaResnet18(SerializableModule):
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        k = 1
        self.backbone = LambdaResNet(LambdaBlock, [2, 2, 2, 2], in_channels=40, k=k)
        self.backbone_dim = 60*k
 
        self.classification_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_dim, num_classes)
        )

    def forward(self, x):
        x.transpose_(2, 1)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.classification_head(x)
        return x
    
    def get_embedding(self, x):
        x.transpose_(2, 1)
        x = x.unsqueeze(1)
        x = self.backbone.get_embedding(x)
        return x
