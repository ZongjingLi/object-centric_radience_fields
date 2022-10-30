import torch
import torch.nn as nn

def make_grid(size = 32):
    return torch.meshgrid()

def render_component(grid,nerf):
    return nerf(grid)

def AffineTransform(x,angle,scale,bias):
    # R[t][sx] + bias
    rot_mat = torch.tensor([
        [torch.cos(angle),-torch.sin(angle)],
        [torch.sin(angle),torch.cos(angle)],
    ])
    return torch.matmul(rot_mat,scale * x) + bias