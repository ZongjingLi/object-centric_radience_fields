import torch
import torch.nn as nn

from config import *
from render import *
from utils  import *

import argparse
train_parser = argparse.ArgumentParser()
train_parser.add_argument("--name", default = "Violet Evergarden")
train_parser.add_argument("--epoch",default = 3000)
train_config = train_parser.parse_args(args = [])

print(train_parser)
