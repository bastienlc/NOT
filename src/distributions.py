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

import random

import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn import datasets


class Sampler:
    def __init__(
        self,
        device="cuda",
    ):
        self.device = device

    def sample(self, size=5):
        pass


class MoonsSampler(Sampler):
    def __init__(self, dim=2, device="cuda"):
        super(MoonsSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2

    def sample(self, batch_size=10):
        batch = (
            datasets.make_moons(n_samples=batch_size, noise=0.1)[0].astype("float32")
            / 7.5
        )

        return torch.tensor(batch, device=self.device)


class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device="cuda"):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)


class Transformer(object):
    def __init__(self, device="cuda"):
        self.device = device


class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device="cuda"):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()

        mean, cov = np.mean(batch, axis=0), np.matrix(np.cov(batch.T))

        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)

        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier), device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(
                self.base_sampler.sample(batch_size), device=self.device
            )
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch
