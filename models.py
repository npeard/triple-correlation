#!/usr/bin/env python

from torch import nn
import torch

act_fn_by_name = {"Tanh": nn.Tanh(), "LeakyReLU": nn.LeakyReLU()}

# Define a linear network
class LinearNet(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, norm=False, Phi_sign=False):
        super(LinearNet, self).__init__()
        self.layers = []
        if not Phi_sign:
            self.layers.append(torch.abs)
        if num_layers > 1:
            self.layers.append(nn.Linear(input_size, hidden_size))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                if norm:
                    self.layers.append(nn.LayerNorm(hidden_size))
            self.layers.append(nn.Linear(hidden_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, output_size))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        pred = self.model(x)
        return pred


# Define a sequential dense network
class SequentialNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size,
                 activation="Tanh", norm=False):
        super(SequentialNN, self).__init__()
        self.layers = []
        # Don't modify inputs before a linear layer, absolute value of inputs
        # is important for learning
        if num_layers > 1:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(act_fn_by_name[activation])
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_size, input_size))
                if norm:
                    self.layers.append(nn.LayerNorm(hidden_size))
                self.layers.append(act_fn_by_name[activation])
            self.layers.append(nn.Linear(hidden_size, output_size))
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.Tanh())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        pred = torch.pi * self.model(x)
        return pred


class LateralBlock(nn.Module):
    def __init__(self, input_size, skip=0, activation="Tanh", norm=False):
        super(LateralBlock, self).__init__()

        if activation == "Tanh":
            self.activate = nn.Tanh()
        elif activation == "LeakyReLU":
            self.activate = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation function")

        if norm:
            self.activate = nn.Sequential(self.activate,
                                          nn.LayerNorm(input_size))

        # networks layers with no connections
        self.layers = []
        for i in range(skip + 1):
            self.layers.append(nn.Linear(input_size, input_size))
            self.layers.append(self.activate)
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.block(x)
        out = torch.add(out, x)
        return out


# Define a dense model with lateral connections
# See https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class LateralNoSkip(nn.Module):
    def __init__(self, input_size, num_layers, num_outputs, activation="Tanh", norm=False):
        super(LateralNoSkip, self).__init__()

        self.layers = []
        # Don't modify inputs before a linear layer, absolute value of inputs
        # is important for learning
        for i in range(num_layers):
            self.layers.append(LateralBlock(input_size, 0, activation, norm))
        self.layers.append(nn.Linear(input_size, num_outputs))
        self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = torch.pi * self.model(x)

        return out


# Define a CNN model that acts on a 2D input and produces the 1D phase output
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
