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

for item in train_config._get_kwargs():
    print("-",item[0],"is",item[1])

def train_render_field(model,train_config = train_config):
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4) # initialize the optimizer

    # set up the experiment dataset
    if train_config.dataset == "sprite3":dataset = Sprite3("train")
    if train_config.dataset == "clevr4": dataset = Clevr4("train")

    # inialize the dataloader according to the train config
    dataloader = DataLoader(dataset, batch_size = train_config.batch_size,shuffle = True)
    for epoch in range(train_config.epoch):
        total_loss = 0 # log the total_loss
        itr = 0
        for sample in dataloader:
            im = sample["image"];itr += 1

            recon = im # calculate the model output
            recon_loss = 0
            comps_loss = 0

            # visualize the training image and output if in vis-iter
            if itr % train_parser.visualize_itr == 0:
                plt.figure("visualize render field")
                plt.subplot(121);plt.cla();plt.imshow(im[0].permute([1,2,0]))
                plt.subplot(122);plt.cla();plt.imshow(recon[0].permute([1,2,0]))
        
            working_loss = recon_loss + comps_loss
            working_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += working_loss # add the working_loss to the total loss
        
        print("epoch:{} loss:{}".format(epoch,total_loss.detach()))

    print("training of neural render field completed.")

if __name__ == "__main__":
    # and let the training begin.
    model = OCRF(config)
    train_render_field(model,train_config)