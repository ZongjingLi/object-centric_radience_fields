import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--im_size",     default = 64)
parser.add_argument("--latent_dim",  default = 100)
parser.add_argument("--concept_dim", default = 100)
parser.add_argument("--space_dim",   default = 2)
parser.add_argument("--nc",          default = 3)
parser.add_argument("--slots",       default = 4)

config = parser.parse_args(args = [])