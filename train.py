import torch
import torch.nn as nn

from config   import *
from render   import *
from utils    import *
from datasets import *

import argparse
train_parser = argparse.ArgumentParser()
train_parser.add_argument("--name", default = "Violet Evergarden")
train_parser.add_argument("--epoch",default = 3000)
train_parser.add_argument("--dataset",default = "sprite3")
train_config = train_parser.parse_args(args = [])

print(train_parser)

def train_render_field(model,train_config = train_config):
    if train_config.dataset == "sprite3":
        dataset = 
