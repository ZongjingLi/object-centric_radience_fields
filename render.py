import torch
import torch.nn as nn

class Nerf(nn.Module):
    def __init__(self,config):
        super().__init__()

    def forward(self,x):
        return x

    def __str__(self):return "neuro radience field"

class PatchDecoder(nn.Module):
    def __init__(self,config):
        super().__init__()

    def forward(self,z):
        # first use the z(100) to decode entity(100) a neural field
        # then  use the z(100) to decode transformation of 
        return z

def PatchLoss(patch,location):
    # minimize the different between the center of patch and the predicted location
    return 0

class OCRF(nn.Module):
    def __init__(self,config):
        # the object centric radience field
        super().__init__()
        self.latent_encoder = None
        self.patch_decoder  = PatchDecoder(config)

    def forward(self,im):
        slot_features = self.latent_encoder(im)
        return im