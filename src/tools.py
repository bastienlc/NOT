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

from .distributions import LoaderSampler

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.datasets import ImageFolder
import torch.nn as nn

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def load_dataset(name, path, img_size=64, batch_size=64, test_ratio=0.1, device='cuda'):
    if name in ['celeba_female', 'celeba_male', 'aligned_anime_faces', 'describable_textures', 'cartoonset100k']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception('Unknown dataset')
        
    if name in ['celeba_female', 'celeba_male']:
        with open('../datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    elif dataset == 'describable_textures':
        idx = np.random.RandomState(seed=0xBADBEEF).permutation(len(dataset))
    else:
        idx = list(range(len(dataset)))
        
    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)