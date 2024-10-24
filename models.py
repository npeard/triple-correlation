#!/usr/bin/env python

from torch import nn
import torch

act_fn_by_name = {"Tanh": nn.Tanh(), "LeakyReLU": nn.LeakyReLU()}


class AbsBlock(nn.Module):
    def __init__(self):
        super(AbsBlock, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class LinearNet(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            hidden_size,
            output_size,
            norm=False
    ):
        super(LinearNet, self).__init__()
        self.layers = []
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
        x_view = x.view(-1, x.size(1)**2)
        phase = self.model(x_view)
        #pred = torch.atan2(torch.sin(phase), torch.cos(phase))
        return phase


class MLP(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size=None,
                 activation="Tanh", norm=False):
        super(MLP, self).__init__()
        self.layers = []
        if num_layers > 1:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(act_fn_by_name[activation])
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                if norm:
                    self.layers.append(nn.LayerNorm(hidden_size))
                self.layers.append(act_fn_by_name[activation])
            self.layers.append(nn.Linear(hidden_size, output_size))
            self.layers.append(act_fn_by_name[activation])
            self.layers.append(nn.Linear(output_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(act_fn_by_name[activation])
            self.layers.append(nn.Linear(output_size, output_size))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x_view = x.view(-1, x.size(1)**2)
        output = self.model(x_view)
        output_view = output.view(-1, x.size(1), x.size(2))
        return output_view
    

class PhaseMLP(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size=None,
                 activation="Tanh", norm=False):
        super(PhaseMLP, self).__init__()
        self.model = MLP(
            input_size=input_size,
            num_layers=num_layers,
            output_size=output_size,
            hidden_size=hidden_size,
            activation=activation,
            norm=norm)

    def forward(self, x):
        x_view = x.view(-1, x.size(1)**2)
        phase = self.model(x_view)
        pred = torch.atan2(torch.sin(phase), torch.cos(phase))
        return pred


class LateralBlock(nn.Module):
    def __init__(self, input_size, skip=0, activation="Tanh", norm=False):
        super(LateralBlock, self).__init__()

        self.activate = act_fn_by_name[activation]

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
        x_view = x.view(-1, x.size(1)**2)
        out = self.block(x_view)
        out = torch.add(out, x_view)
        return out


# Define a dense model with lateral connections
# See https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class LateralNoSkip(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_outputs,
            activation="Tanh",
            norm=False):
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
        x_view = x.view(-1, x.size(1)**2)
        out = torch.pi * self.model(x_view)

        return out


class ConvolutionBlock(nn.Module):
    def __init__(
            self,
            output_channels,
            num_layers,
            activation="Tanh",
            kernel_size=3,
            dropout_rate=0):
        super(ConvolutionBlock, self).__init__()

        self.activate = act_fn_by_name[activation]

        # network layers with no connections
        self.layers = []
        self.layers.append(
            nn.Conv2d(
                1,
                16,
                kernel_size=kernel_size,
                padding='same'))
        self.layers.append(self.activate)
        if dropout_rate > 0:
            self.layers.append(nn.Dropout(dropout_rate))
        for i in range(num_layers - 1):
            # self.layers.append(nn.MaxPool2d(kernel_size=2))
            self.layers.append(nn.Conv2d(
                16 * (i + 1), 16 * (i + 2), kernel_size=kernel_size, padding='same'))
            self.layers.append(self.activate)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(
            nn.Conv2d(
                16 * num_layers,
                output_channels,
                kernel_size=kernel_size,
                padding='same'))
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        x_view = x.view(-1, 1, x.size(1), x.size(2))
        # Applying global average pooling
        out = torch.mean(self.block(x_view), dim=1, keepdim=True)
        # Keeping dimensions to remain compatible with the fully connected
        # block
        return out


# Define a CNN model that acts on a 2D input and produces the 1D phase output
# use the same number of channels in each layer with "same" padding
# computationally inefficient, but easy to implement and preserves edge
# information
class BottleCNN(nn.Module):
    def __init__(
            self,
            input_size,
            num_conv_layers,
            num_layers,
            kernel_size,
            output_size,
            hidden_size=None,
            activation="Tanh",
            norm=False,
            dropout_rate=0):
        super(BottleCNN, self).__init__()
        self.conv_block = ConvolutionBlock(
            output_channels=output_size,
            num_layers=num_conv_layers,
            kernel_size=kernel_size,
            activation=activation,
            dropout_rate=dropout_rate)
        self.fc_block = MLP(
            input_size=input_size,
            num_layers=num_layers,
            output_size=output_size,
            hidden_size=hidden_size,
            activation=activation,
            norm=norm)

    def forward(self, x):
        x_view = x.view(-1, 1, x.size(1), x.size(2))
        out = self.conv_block(x_view)
        # flatten the output of the convolutional block for the fully connected block
        # (batch_size, channels, dim1, dim2) -> (batch_size, channels*dim1*dim2)
        phase = self.fc_block(
            out.view(
                out.size(0),
                out.size(1) *
                out.size(2) *
                out.size(3)))
        # Don't need to do atan2 because it is done in the MLP block
        # pred = torch.atan2(torch.sin(phase), torch.cos(phase))
        return phase
