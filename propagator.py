import torch
import torch.nn as nn

class NOPropNet(nn.Module):
    def __init__(self,config):
        # this is meant to construct a neuro-object propagation network
        # thinking about augument this in the field of neuro objects https://yilundu.github.io/podnet/
        super().__init__()

    def forward(self,x):
        # http://propnet.csail.mit.edu/ consider this one
        # input structure should be a set of node (probably with edges)
        # if there are no input edges, try to construct some edges
        # some principles during the design follows the work of the pod net
        # https://yilundu.github.io/podnet/
        x = x
        # for each input node, it should have the structure like [1,(e,dim)]
        # which represents a neuro radience field that has the concept embedding e, and location at
        # r, and rotation,scaling according to the affine transformation

        # Differentiable Physics Simulation of Dynamics Augmented Neural Objects
        # https://arxiv.org/abs/2210.09420
        # the output of the forward step should be to modfiy the spatial tranformation r,A,s according 
        # to the dynamic rules (and also change the entity embedding for any change in attributes and shapes)
        return x