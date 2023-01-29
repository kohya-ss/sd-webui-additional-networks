import os
from modules import paths


def preload(parser):
    parser.add_argument("--addnet-max-model-count", type=int, help="The maximum number of additional network model can be used.", default=5)