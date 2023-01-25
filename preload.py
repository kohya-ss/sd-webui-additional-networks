import os
from modules import paths


def preload(parser):
    parser.add_argument("--max-lora-count", type=int, help="The maximum number of LoRA model can be used.", default=5)