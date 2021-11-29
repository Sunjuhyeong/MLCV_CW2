import os

import torch
import random
import numpy as np

import shutil
from glob import glob


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(images):
    out = (images + 1) / 2
    return out.clamp(0, 1)


def make_ckpt_directory(train_config):
    path = f"checkpoints/{train_config['name']}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(f'{path}/images/', exist_ok=True)
    os.makedirs(f'{path}/weights/', exist_ok=True)
    train_config['ckpt_path'] = path

    os.makedirs(f'{path}/config/', exist_ok=True)
    for file in glob("config/*.yaml"):
        shutil.copy(file, f"{path}/{file}")

    for file in glob("*.py"):
        shutil.copy(file, f"{path}/{file}")

