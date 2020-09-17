#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class CouplingLayer(nn.Module):
    def __init__(self, dim, key):
        super().__init__()

        self.key = key

        # coupling functionには元論文のMNIST用の設定を採用
        dim_h = 1000
        n_hidden = 5
        self.m = nn.Sequential(
            nn.Sequential(nn.Linear(int(dim / 2), dim_h), nn.ReLU()),
            *[
                nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                for _ in range(n_hidden - 1)
            ],
            nn.Sequential(nn.Linear(dim_h, int(dim / 2))),
        )

        # あまり重要ではないかもしれないが、一応ちゃんとした初期化をかけておく
        for i in range(len(self.m) - 1):
            nn.init.kaiming_normal_(
                self.m[i][0].weight.data, mode="fan_in", nonlinearity="relu"
            )
            nn.init.zeros_(self.m[i][0].bias.data)
        nn.init.xavier_normal_(self.m[-1][0].weight.data)
        nn.init.zeros_(self.m[-1][0].bias.data)

    def forward(self, data_in, inverse=False):

        data_out = torch.empty_like(data_in)

        # additive coupling law
        if not inverse:
            x_I1 = data_in[:, 1::2]
            x_I2 = data_in[:, 0::2]

            if self.key == "odd":
                y_I1 = x_I1
                y_I2 = x_I2 + self.m(x_I1)
            else:
                y_I1 = x_I1 + self.m(x_I2)
                y_I2 = x_I2

            data_out[:, 1::2] = y_I1
            data_out[:, 0::2] = y_I2

        if inverse:
            y_I1 = data_in[:, 1::2]
            y_I2 = data_in[:, 0::2]

            if self.key == "odd":
                x_I1 = y_I1
                x_I2 = y_I2 - self.m(y_I1)
            else:
                x_I1 = y_I1 - self.m(y_I2)
                x_I2 = y_I2

            data_out[:, 1::2] = x_I1
            data_out[:, 0::2] = x_I2

        log_det_J = torch.zeros((data_out.size(0), 1)).to(device)

        return data_out, log_det_J


class RescalingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Parameter(torch.zeros((dim,)))

    def forward(self, x, inverse=False):

        # ArchitextureにはS_{ii}というscaling matrixを用いると書いてあったが、
        # Experimentsにはexp(s)を各要素に掛ける旨が書いてあったのでそれを採用。
        # おそらく非負条件を満たすためにそうしているのだろう。
        if not inverse:
            out = x * torch.exp(self.s)
        if inverse:
            out = x / torch.exp(self.s)

        log_det_J = torch.sum(self.s).expand((out.size(0), 1))

        return out, log_det_J


class NICE(nn.Module):
    def __init__(self, shape):
        super().__init__()

        dim = np.prod(shape)

        self.flatten_layer = Reshape(-1)
        self.layers = nn.Sequential(
            CouplingLayer(dim, key="odd"),
            CouplingLayer(dim, key="even"),
            CouplingLayer(dim, key="odd"),
            CouplingLayer(dim, key="even"),
            RescalingLayer(dim),
        )
        self.to_img_layer = Reshape(*shape)

    def forward(self, x, inverse=False):

        sum_log_det_J = 0
        out = self.flatten_layer(x)

        if not inverse:
            for i in range(len(self.layers)):
                out, log_det_J = self.layers[i](out)
                sum_log_det_J += log_det_J

        elif inverse:
            for i in reversed(range(len(self.layers))):
                out, log_det_J = self.layers[i](out, inverse)
                sum_log_det_J += log_det_J

            out = self.to_img_layer(out)

        return out, sum_log_det_J

    @staticmethod
    def loss_func(z, sum_log_det_J):

        log_p_Z_z = torch.sum(-0.5 * (z ** 2 + np.log(2 * np.pi)), dim=1, keepdim=True)

        log_p_X_x = log_p_Z_z + sum_log_det_J
        loss = -torch.mean(log_p_X_x)
        return loss
