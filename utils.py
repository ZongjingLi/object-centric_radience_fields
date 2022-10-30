import torch
import torch.nn as nn

def patch_center(patch):
    # input patch shape [b,im_size,im_size,3]
    if patch.shape[-1] != 1:patch = torch.max(patch,-1)[0].float()
    grid = make_grid(patch.shape[1]).unsqueeze(0)
    total_mass = patch.sum(1).sum(1)
    patch_dist = (patch * grid).sum(1).sum(1)
    return patch_dist/total_mass

def make_grid(im_size = 32):
    x = torch.linspace(0, 1, im_size);y = torch.linspace(0, 1, im_size)
    x_grid,y_grid = torch.meshgrid(x, y)
    return torch.stack([x_grid,y_grid],-1)

def render_component(grid,nerf,aff):
    return nerf(aff(grid)) # use the affine transformation to change the grid frist then use the nerf to render the image

def AffineTransform(x,angle,scale,bias):
    # R[t][sx] + bias
    rot_mat = torch.tensor([
        [torch.cos(angle),-torch.sin(angle)],
        [torch.sin(angle),torch.cos(angle)],
    ])
    return torch.matmul(rot_mat,scale * x) + bias