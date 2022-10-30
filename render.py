import torch
import torch.nn as nn
from moic.mklearn.nn.functional_net import FCBlock
from utils import *

class Nerf(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dim  = config.space_dim
        self.latent_dim = config.concept_dim
        self.nerf = FCBlock(132,4,self.dim + self.latent_dim,config.nc)
        self.b = torch.randn([1,32,32,3])
    def forward(self,z,x):return self.nerf(10 * torch.cat([x,z],-1)) * 0.5 + 0.5

    def __str__(self):return "{} dim neuro radience field".format(self.dim)

class PatchDecoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config= config
        latent_dim = config.latent_dim
        self.spatial_feature = FCBlock(132,2,latent_dim,latent_dim)
        # first decoder the spatial feature
        self.scale_decoder = FCBlock(132,3,latent_dim,1) # decode the raw scale
        self.angle_decoder = FCBlock(132,3,latent_dim,1) # decode raw angle
        self.shift_decoder = FCBlock(132,3,latent_dim,config.space_dim)
        # from spatial features decode the affine parameters
        self.concept_decoder = FCBlock(132,3,latent_dim,config.concept_dim)

    def forward(self,z):
        # first use the z(100) to decode entity(100) a neural field
        # then  use the z(100) to decode the affine transformation A s + b

        # input latent-z shape [b,k,z]
        spatial_latent = self.spatial_feature(z)
        # repre the spatial feature use a MLP
        scale = torch.sigmoid(self.scale_decoder(spatial_latent) * 3)
        angle = self.angle_decoder(spatial_latent) * torch.pi * 2
        shift = self.shift_decoder(spatial_latent) * 0.5 #[-1/2,1/2]
        # decode the entity in the box concept space
        grid = make_grid(self.config.im_size)
        concept_latent = self.concept_decoder(z)

        return {"entity":concept_latent,
                "scale":scale,
                "angle":angle,
                "shift":shift}

def PatchLoss(patch,locations):
    # minimize the different between the center of patch and the predicted location
    dist_center = patch_center(patch)
    return torch.pow(dist_center-locations,2).mean()

class OCRF(nn.Module):
    def __init__(self,config):
        # the object centric radience field
        super().__init__()
        self.latent_encoder = SlotAttention(config.slots,config.latent_dim,slot_dim = config.latent_dim)
        self.patch_decoder  = PatchDecoder(config)
        self.render_field   = Nerf(config)

    def forward(self,im):
        slot_features = self.latent_encoder(im)
        print(slot_features.shape)
        patch_infos   = self.patch_decoder(slot_features)

        # decode the nerf fields components
        grids = None
        entities  = patch_infos["entity"]
        scales    = patch_infos["scale"]
        angles    = patch_infos["angle"]
        shifts    = patch_infos["shift"]
        print(entities.shape)
        affine_grids = AffineTransform(grids,angles,scales,shifts)
        components = self.render_field(affine_grids)
        return components


class SlotAttention(nn.Module):
    def __init__(self,num_slots,in_dim=64,slot_dim=64,iters=3,eps=1e-8,hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots-1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots-1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma_bg)
        
        self.kslots_mu = nn.Parameter(torch.randn(1,num_slots,slot_dim))
        self.kslots_logsigma = nn.Parameter(torch.randn(1,num_slots,slot_dim))

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self,feat,num_slots = None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.repeat(B,1,1)#.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().repeat(B,1,1)#.expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
        
        #mu_bg = self.slots_mu.expand(B, 1, -1)
        #sigma_bg = self.slots_logsigma.exp().expand(B, 1, -1)
        #slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn