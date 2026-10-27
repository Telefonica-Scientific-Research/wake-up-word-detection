# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import torch
import torch.nn.functional as F
import numpy as np
def affine(spec, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0]):
    if len(spec.size()) == 2:
        spec = spec.unsqueeze(0).unsqueeze(0) 
    elif len(spec.size()) == 3:  
        spec = spec.unsqueeze(0)  
    batch_size, channels, height, width = spec.size()
    transform_matrix = torch.tensor([[scale, shear[0], translate[0]],
                                     [shear[1], scale, translate[1]]]).float().unsqueeze(0)
    transform_matrix = transform_matrix.to(spec.device).expand(batch_size, -1, -1)
    grid = F.affine_grid(transform_matrix, torch.Size([batch_size, channels, height, width]), align_corners=False)
    transformed_spec = F.grid_sample(spec, grid, mode='bilinear', align_corners=False)
    return transformed_spec.squeeze(0).squeeze(0)

def rotate(spec, degrees):
    if spec.dim() == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)
    elif spec.dim() == 3:
        spec = spec.unsqueeze(1)   
    batch_size, channels, height, width = spec.size()
    angle_rad = torch.tensor(degrees * torch.pi / 180.0).float() 
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    transform_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                     [sin_a, cos_a, 0]]).float().unsqueeze(0)
    transform_matrix = transform_matrix.to(spec.device).expand(batch_size, -1, -1)
    grid = F.affine_grid(transform_matrix, torch.Size([batch_size, channels, height, width]), align_corners=False)
    rotated_spec = F.grid_sample(spec, grid, mode='bilinear', align_corners=False)
    if rotated_spec.size(1) == 1:  
        rotated_spec = rotated_spec.squeeze(1)
    if rotated_spec.size(0) == 1: 
        rotated_spec = rotated_spec.squeeze(0)

    return rotated_spec

def adjust_brightness(spec, factor):
    return torch.clamp(spec * factor, 0.0, 1.0)

def adjust_contrast(spec, factor):
    mean = spec.mean()
    return torch.clamp((spec - mean) * factor + mean, 0.0, 1.0)

def adjust_saturation(spec, factor):
    grayscale = spec.mean(dim=0, keepdim=True)
    return torch.clamp(grayscale + (spec - grayscale) * factor, 0.0, 1.0)
