#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import NICE
from util import plot_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1024
image_shape = (1, 28, 28)
(col_plot, row_plot) = (5, 4)
n_test_plot = col_plot * 4


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    trainloader = DataLoader(
        datasets.MNIST(
            root="./datasets/", train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = NICE(image_shape).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-4, weight_decay=0
    )

    z_test = torch.randn((n_test_plot, *image_shape)).to(device)

    for i in range(1000):

        # ===== train =====
        model.train()

        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for x, _ in trainloader:

                x = x.to(device)

                z, sum_log_det_J = model(x)
                loss = model.loss_func(z, sum_log_det_J)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(x.shape[0])

        # ===== test =====
        model.eval()

        img_for_plot, _ = model(z_test, inverse=True)

        plot_images(img_for_plot, col=col_plot, row=row_plot)
        plt.savefig("img/figure_" + str(i) + ".png")


if __name__ == "__main__":
    main()
