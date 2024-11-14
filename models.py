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
        # TODO: add conditional reshaping here
        output_view = output.view(-1, x.size(1), x.size(2))
        return output_view
    
    
class ImplicitMultiMLP(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size=None,
                 activation="Tanh", norm=False):
        super(ImplicitMultiMLP, self).__init__()
        self.layers = []
        self.mlp = MLP(input_size=input_size, num_layers=num_layers,
                  output_size=input_size, hidden_size=hidden_size,
                  activation=activation, norm=norm)
        self.lin = LinearNet(input_size=input_size, num_layers=1,
                        output_size=output_size, hidden_size=hidden_size,
                        norm=norm)
        
    def forward(self, x):
        #x_view = x.view(-1, x.size(1)**2)
        sign_prob = self.mlp(x)
        sign = nn.Tanh()(sign_prob)
        #sign_view = sign.view(-1, x.size(1), x.size(2))
        Phi = x * sign
        phase = self.lin(Phi)
        # This atan2 operation is very unhelpful for optimization, why?
        #pred = torch.atan2(torch.sin(phase), torch.cos(phase))
        return phase

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


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_heads=4):
        super().__init__()
        
        # Initial embedding of individual features
        self.feature_embedding = nn.Linear(1, hidden_size)
        
        # Multi-head self attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Process attention output
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Global pooling and classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            #nn.Tanh()  # Output in [-1, 1] for sign classification
        )
    
    def forward(self, x):
        x_view = x.view(-1, x.size(1)**2)
        batch_size, num_features = x_view.shape
        
        # Reshape and embed each feature
        x_view = x_view.view(batch_size, num_features, 1)
        embeddings = self.feature_embedding(x_view)
        
        # Self attention to capture correlations
        attention_out, _ = self.self_attention(
            embeddings, embeddings, embeddings
        )
        
        # Process attention outputs
        processed = self.post_attention(attention_out)
        
        # Global mean pooling across features
        pooled = torch.mean(processed, dim=1)
        
        # Final classification
        output = self.classifier(pooled)
        output_view = output.view(-1, x.size(1), x.size(2))
        
        return output_view


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
