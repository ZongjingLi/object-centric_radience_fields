import torch
import torch.nn as nn

from config   import *
from render   import *
from utils    import *
from datasets import *

import matplotlib.pyplot as plt

import argparse
train_parser = argparse.ArgumentParser()
train_parser.add_argument("--name",          default = "Violet Evergarden")
train_parser.add_argument("--epoch",         default = 3000)
train_parser.add_argument("--dataset",       default = "sprite3")
train_parser.add_argument("--batch_size",    default = 4)
train_parser.add_argument("--visualize_itr", default = 3)
train_config = train_parser.parse_args(args = [])

for item in train_config._get_kwargs():print("-",item[0],"is",item[1])

def train_render_field(model,train_config = train_config):
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4) # initialize the optimizer

    # set up the experiment dataset
    if train_config.dataset == "sprite3":dataset = Sprite3("train")
    if train_config.dataset == "clevr4": dataset = Clevr4("train")

    # inialize the dataloader according to the train config
    dataloader = DataLoader(dataset, batch_size = train_config.batch_size,shuffle = True)
    # define the loss function 
    loss_func  = torch.nn.MSELoss(reduce = "mean")
    loss_history = []
    for epoch in range(train_config.epoch):
        total_loss = 0 # log the total_loss
        itr = 0
        for sample in dataloader:
            im = sample["image"];itr += 1 # [b,64,64,3]

            results = im 
            recon   = results["recon"] # calculate the model output [b,64,64,3]
            comps   = results["comps"] # calculate the components of reconstruction [b,n,64,64,3]
            cents   = results["cents"] # calculate the predict center of the patch  [b,n,3]

            recon_loss = loss_func(recon,im)
            comps_loss = PatchLoss(comps,cents) # whether the center of the image is aligned with 

            # visualize the training image and output if in vis-iter
            if itr % train_parser.visualize_itr == 0:
                plt.figure("visualize render field")
                plt.subplot(121);plt.cla();plt.imshow(im[0].permute([1,2,0]))
                plt.subplot(122);plt.cla();plt.imshow(recon[0].permute([1,2,0]))
            
            # total working loss constists of the reconstruction loss and the patch alignment loss
            working_loss = recon_loss + comps_loss
            working_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # add the working loss into the total loss 
            total_loss += working_loss.detach() # add the working_loss to the total loss
        
        print("epoch:{} loss:{}".format(epoch,total_loss.detach()))
        loss_history.appenda(total_loss)

    print("training of neural render field completed.")

if __name__ == "__main__":
    # and let the training begin.
    model = OCRF(config)
    train_render_field(model,train_config)

    