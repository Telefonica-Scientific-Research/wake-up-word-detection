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
import torch.nn.functional as F


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


class LambdaBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.ModuleList([LambdaLayer1D(planes, planes)])
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool1d(kernel_size=3, stride=1, padding=1))
        
        self.conv2.append(nn.BatchNorm1d(planes))
        self.conv2.append(nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(*self.conv2)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=False)
        return out


class LambdaResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, k=1):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # There are 4 layers in resnet, each layer has it's number of blocks
        self.layer1 = self._make_layer(block, int(24*k), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(36*k), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(48*k), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(60*k), num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # In case the MEL has a 1 channel
        if x.shape[1] == 1 and len(x.shape) > 3:
            x = torch.squeeze(x, dim=1)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    
    def get_embedding(self, x):
        # In case the MEL has a 1 channel
        if x.shape[1] == 1 and len(x.shape) > 3:
            x = torch.squeeze(x, dim=1)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out