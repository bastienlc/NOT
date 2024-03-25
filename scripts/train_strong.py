"""
Code adapted from https://github.com/iamalexkorotin/NeuralOptimalTransport

MIT License

Copyright (c) 2023 Alexander

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys

sys.path.append("..")
sys.path.append("../..")

import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb  # <--- online logging of the results

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
from tqdm import tqdm

from src.plotters import plot_images, plot_random_images
from src.resnet2 import ResNet_D
from src.tools import fig2img  # for wandb
from src.tools import freeze, load_dataset, unfreeze, weights_init_D
from src.unet import UNet

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

DEVICE_IDS = [0]

DATASET1, DATASET1_PATH = "celeba_female", "../../data/img_align_celeba"
DATASET2, DATASET2_PATH = "cartoonset100k", "../../data/cartoonset100k"

T_ITERS = 10
f_LR, T_LR = 1e-4, 1e-4
IMG_SIZE = 128

BATCH_SIZE = 64

PLOT_INTERVAL = 100
COST = "mse"  # Mean Squared Error
CPKT_INTERVAL = 2000
MAX_STEPS = 100001
SEED = 0x000000

EXP_NAME = f"{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}"
OUTPUT_PATH = "../checkpoints/{}/{}_{}_{}/".format(COST, DATASET1, DATASET2, IMG_SIZE)

config = dict(
    DATASET1=DATASET1,
    DATASET2=DATASET2,
    T_ITERS=T_ITERS,
    f_LR=f_LR,
    T_LR=T_LR,
    BATCH_SIZE=BATCH_SIZE,
)

assert torch.cuda.is_available()
torch.cuda.set_device(f"cuda:{DEVICE_IDS[0]}")
torch.manual_seed(SEED)
np.random.seed(SEED)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

X_sampler, X_test_sampler = load_dataset(
    DATASET1, DATASET1_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE
)
Y_sampler, Y_test_sampler = load_dataset(
    DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE
)

torch.cuda.empty_cache()
gc.collect()

f = ResNet_D(IMG_SIZE, nc=3).cuda()
f.apply(weights_init_D)

T = UNet(3, 3, base_factor=48).cuda()

if len(DEVICE_IDS) > 1:
    T = nn.DataParallel(T, device_ids=DEVICE_IDS)
    f = nn.DataParallel(f, device_ids=DEVICE_IDS)

print("T params:", np.sum([np.prod(p.shape) for p in T.parameters()]))
print("f params:", np.sum([np.prod(p.shape) for p in f.parameters()]))

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)
X_fixed = X_sampler.sample(10)
Y_fixed = Y_sampler.sample(10)
X_test_fixed = X_test_sampler.sample(10)
Y_test_fixed = Y_test_sampler.sample(10)

wandb.init(name=EXP_NAME, project="not-experiments", config=config)

T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)

for step in tqdm(range(MAX_STEPS)):
    # T optimization
    unfreeze(T)
    freeze(f)
    for t_iter in range(T_ITERS):
        T_opt.zero_grad()
        X = X_sampler.sample(BATCH_SIZE)
        T_X = T(X)
        if COST == "mse":
            T_loss = F.mse_loss(X, T_X).mean() - f(T_X).mean()
        else:
            raise Exception("Unknown COST")
        T_loss.backward()
        T_opt.step()
    del T_loss, T_X, X
    gc.collect()
    torch.cuda.empty_cache()

    # f optimization
    freeze(T)
    unfreeze(f)
    X = X_sampler.sample(BATCH_SIZE)
    with torch.no_grad():
        T_X = T(X)
    Y = Y_sampler.sample(BATCH_SIZE)
    f_opt.zero_grad()
    f_loss = f(T_X).mean() - f(Y).mean()
    f_loss.backward()
    f_opt.step()
    wandb.log({f"f_loss": f_loss.item()}, step=step)
    del f_loss, Y, X, T_X
    gc.collect()
    torch.cuda.empty_cache()

    if step % PLOT_INTERVAL == 0:
        print("Plotting")

        fig, axes = plot_images(X_fixed, Y_fixed, T)
        wandb.log({"Fixed Images": [wandb.Image(fig2img(fig))]}, step=step)
        plt.close(fig)

        fig, axes = plot_random_images(X_sampler, Y_sampler, T)
        wandb.log({"Random Images": [wandb.Image(fig2img(fig))]}, step=step)
        plt.close(fig)

        fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
        wandb.log({"Fixed Test Images": [wandb.Image(fig2img(fig))]}, step=step)
        plt.close(fig)

        fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
        wandb.log({"Random Test Images": [wandb.Image(fig2img(fig))]}, step=step)
        plt.close(fig)

    if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
        freeze(T)
        torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f"{SEED}_{step}.pt"))

    gc.collect()
    torch.cuda.empty_cache()
