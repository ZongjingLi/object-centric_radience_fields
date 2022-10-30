import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--im_size",     default = 32)
parser.add_argument("--latent_dim",  default = 100)
parser.add_argument("--concept_dim", default = 100)
parser.add_argument("--space_dim",   default = 2)

config = parser.parse_args(args = [])