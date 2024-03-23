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

import matplotlib.pyplot as plt
import ot
import torch
from matplotlib import collections as mc

from .tools import freeze


def plot_bar_and_stochastic_2D(X_sampler, Y_sampler, T, ZD, Z_STD, plot_discrete=True):
    DIM = 2
    freeze(T)

    DISCRETE_OT = 1024

    PLOT_X_SIZE_LEFT = 64
    PLOT_Z_COMPUTE_LEFT = 256

    PLOT_X_SIZE_RIGHT = 32
    PLOT_Z_SIZE_RIGHT = 4

    assert PLOT_Z_COMPUTE_LEFT >= PLOT_Z_SIZE_RIGHT
    assert PLOT_X_SIZE_LEFT >= PLOT_X_SIZE_RIGHT
    assert DISCRETE_OT >= PLOT_X_SIZE_LEFT

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.2),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    for i in range(2):
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(-2.5, 2.5)

    axes[0].set_title(
        r"Map $x\mapsto \overline{T}(x)=\int_{\mathcal{Z}}T(x,z)d\mathbb{S}(z)$",
        fontsize=22,
        pad=10,
    )
    axes[1].set_title(r"Stochastic map $x\mapsto T(x,z)$", fontsize=20, pad=10)
    axes[2].set_title(r"DOT map $x\mapsto \int y d\pi^{*}(y|x)$", fontsize=18, pad=10)

    # Computing and plotting discrete OT bar map
    X, Y = X_sampler.sample(DISCRETE_OT), Y_sampler.sample(DISCRETE_OT)

    if plot_discrete:
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        pi = ot.weak.weak_optimal_transport(X_np, Y_np)
        T_X_bar_np = pi @ Y_np * len(X)

        lines = list(zip(X_np[:PLOT_X_SIZE_LEFT], T_X_bar_np[:PLOT_X_SIZE_LEFT]))
        lc = mc.LineCollection(lines, linewidths=1, color="black")
        axes[2].add_collection(lc)
        axes[2].scatter(
            X_np[:PLOT_X_SIZE_LEFT, 0],
            X_np[:PLOT_X_SIZE_LEFT, 1],
            c="darkseagreen",
            edgecolors="black",
            zorder=2,
            label=r"$x\sim\mathbb{P}$",
        )
        axes[2].scatter(
            T_X_bar_np[:PLOT_X_SIZE_LEFT, 0],
            T_X_bar_np[:PLOT_X_SIZE_LEFT, 1],
            c="slateblue",
            edgecolors="black",
            zorder=2,
            label=r"$\overline{T}(x)$",
            marker="v",
        )
        axes[2].legend(fontsize=16, loc="lower right", framealpha=1)

    # Our method results
    with torch.no_grad():
        X = X[:PLOT_X_SIZE_LEFT].reshape(-1, 1, DIM).repeat(1, PLOT_Z_COMPUTE_LEFT, 1)
        Y = Y[:PLOT_X_SIZE_LEFT]

        Z = (
            torch.randn(PLOT_X_SIZE_LEFT, PLOT_Z_COMPUTE_LEFT, ZD, device="cuda")
            * Z_STD
        )
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = (
            T(XZ.flatten(start_dim=0, end_dim=1))
            .permute(1, 0)
            .reshape(DIM, -1, PLOT_Z_COMPUTE_LEFT)
            .permute(1, 2, 0)
        )

    X_np = X[:, 0].cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ.cpu().numpy()

    lines = list(zip(X_np[:PLOT_X_SIZE_LEFT], T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT]))
    lc = mc.LineCollection(lines, linewidths=1, color="black")
    axes[0].add_collection(lc)
    axes[0].scatter(
        X_np[:PLOT_X_SIZE_LEFT, 0],
        X_np[:PLOT_X_SIZE_LEFT, 1],
        c="darkseagreen",
        edgecolors="black",
        zorder=2,
        label=r"$x\sim\mathbb{P}$",
    )
    axes[0].scatter(
        T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 0],
        T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 1],
        c="tomato",
        edgecolors="black",
        zorder=2,
        label=r"$\overline{T}(x)$",
        marker="v",
    )
    axes[0].legend(fontsize=16, loc="lower right", framealpha=1)

    lines = []
    for i in range(PLOT_X_SIZE_RIGHT):
        for j in range(PLOT_Z_SIZE_RIGHT):
            lines.append((X_np[i], T_XZ_np[i, j]))
    lc = mc.LineCollection(lines, linewidths=0.5, color="black")
    axes[1].add_collection(lc)
    axes[1].scatter(
        X_np[:PLOT_X_SIZE_RIGHT, 0],
        X_np[:PLOT_X_SIZE_RIGHT, 1],
        c="darkseagreen",
        edgecolors="black",
        zorder=2,
        label=r"$x\sim\mathbb{P}$",
    )
    axes[1].scatter(
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 0].flatten(),
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 1].flatten(),
        c="wheat",
        edgecolors="black",
        zorder=3,
        label=r"$T(x,z)$",
    )
    axes[1].legend(fontsize=16, loc="lower right", framealpha=1)

    fig.tight_layout()
    return fig, axes


def plot_generated_2D(X_sampler, Y_sampler, T, ZD, Z_STD):
    DIM = 2
    freeze(T)

    PLOT_SIZE = 512
    X = X_sampler.sample(PLOT_SIZE).reshape(-1, 1, DIM).repeat(1, 1, 1)
    Y = Y_sampler.sample(PLOT_SIZE)

    with torch.no_grad():
        Z = torch.randn(PLOT_SIZE, 1, ZD, device="cuda") * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = (
            T(XZ.flatten(start_dim=0, end_dim=1))
            .permute(1, 0)
            .reshape(DIM, -1, 1)
            .permute(1, 2, 0)
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True, dpi=150)

    X_np = X[:, 0].cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ[:, 0].cpu().numpy()

    for i in range(3):
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(-2.5, 2.5)
        axes[i].grid(True)

    axes[0].scatter(X_np[:, 0], X_np[:, 1], c="darkseagreen", edgecolors="black")
    axes[1].scatter(Y_np[:, 0], Y_np[:, 1], c="peru", edgecolors="black")
    axes[2].scatter(T_XZ_np[:, 0], T_XZ_np[:, 1], c="wheat", edgecolors="black")

    axes[0].set_title(r"Input $x\sim\mathbb{P}$", fontsize=22, pad=10)
    axes[1].set_title(r"Target $y\sim\mathbb{Q}$", fontsize=22, pad=10)
    axes[2].set_title(
        r"Fitted $T(x,z)_{\#}(\mathbb{P}\times\mathbb{S})$", fontsize=22, pad=10
    )

    fig.tight_layout()
    return fig, axes

def plot_images(X, Y, T):
    freeze(T);
    with torch.no_grad():
        T_X = T(X)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)

