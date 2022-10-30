import torch
import torch.nn as nn

from config import *
from render import *

from utils import *

from datasets import *

model = OCRF(config)

im = torch.randn([3,32,32,3])
#outputs = model(im)

import matplotlib.pyplot as plt

nf = Nerf(config)

grid = make_grid(64)

t = torch.tensor(0.0)
s = torch.tensor(1.0)
shift = torch.tensor([0.5,0.5])
concept = torch.randn([1,100])

sprite3 = Sprite3("train",resolution = (64,64))
ft = sprite3[1]["image"].permute([1,2,0]).unsqueeze(0)
optim = torch.optim.Adam(nf.parameters(),lr = 1e-4)
for i in range(1000):
    affine_grid = AffineTransform(grid,t,s,shift).unsqueeze(0)
    clat = concept.unsqueeze(1).unsqueeze(1).repeat([1,64,64,1])

    comp = nf(affine_grid,clat)
    plt.imshow(comp[0].detach());plt.pause(0.01);plt.cla()
    loss = torch.nn.MSELoss()(comp,ft)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print(i,loss.detach())

plt.ion()
for i in range(1000):
    affine_grid = AffineTransform(grid,t,s,shift).unsqueeze(0)
    clat = concept.unsqueeze(1).unsqueeze(1).repeat([1,64,64,1])

    comp = nf(affine_grid,clat)
    plt.imshow(comp[0].detach());plt.pause(0.01);plt.cla()
    t += 0.1
    print(i)

for i in range(1000):
    affine_grid = AffineTransform(grid,t,s,shift).unsqueeze(0)
    clat = concept.unsqueeze(1).unsqueeze(1).repeat([1,64,64,1])

    comp = nf(affine_grid,clat)
    plt.imshow(comp[0].detach());plt.pause(0.01);plt.cla()
    shift -= 0.05
    s += 0.1
    print(i)