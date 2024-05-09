#!/usr/bin/env python

from torch import nn
import torch


# Define a single layer linear model
class SingleLinear(nn.Module):
    def __init__(self, input_size, num_outputs):
        super(SingleLinear, self).__init__()
        self.fc = nn.Linear(input_size, num_outputs)

    def forward(self, x):
        out = self.fc(x)
        return out


# Define a multilayer linear model
class MultiLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(MultiLinear, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# Define a single layer perceptron model
class SinglePerceptron(nn.Module):
    def __init__(self, input_size, num_outputs):
        super(SinglePerceptron, self).__init__()
        self.fc = nn.Linear(input_size, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = torch.pi * self.tanh(self.fc(x))
        return out


# Define a sequential dense model
class SequentialNN(nn.Module):
    def __init__(self, input_size, num_layers, num_outputs, activation="Tanh", norm=False):
        super(SequentialNN, self).__init__()
        self.layers = []
        # Don't modify inputs before a linear layer, absolute value of inputs
        # is important for learning
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size, input_size))
            if norm:
                self.layers.append(nn.LayerNorm(input_size))

            if activation == "Tanh":
                self.layers.append(nn.Tanh())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            else:
                raise ValueError("Invalid activation function")
        self.layers.append(nn.Linear(input_size, num_outputs))
        self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = torch.pi * self.model(x)

        return out


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
