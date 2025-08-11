"""Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# source: https://github.com/karpathy/nanoGPT/blob/master/model.py


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support bias=False."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inp, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = config.is_causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning('Using slow attention. Flash Attention requires PyTorch >= 2.0')
            # causal mask to ensure attention is only applied to the left in the input
            # sequence
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # logger.debug('Using flash attention')
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Chomp1d(nn.Module):
    """Remove padding at the end of the sequence."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single block of temporal convolutions with dilation."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        activation_fn: nn.Module = None,
    ):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.LeakyReLU()
            
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = activation_fn
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = activation_fn
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation2,
            self.dropout2,
        )

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.final_activation = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)


class RegressionTCN(nn.Module):
    """TCN-based regression head that takes data with n_embd channels of length
    in_seq_len and applies temporal convolutional layers followed by average pooling
    to get the final output.
    """

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.config = config

        # Get activation function
        act_fn_map = {
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU()
        }
        activation_fn = act_fn_map.get(config.reg_tcn_activation, nn.LeakyReLU())

        # Build TCN layers
        layers = []
        num_levels = len(config.reg_tcn_num_channels)
        for i in range(num_levels):
            dilation_size = config.reg_tcn_dilation_base ** i
            in_channels = config.n_embd if i == 0 else config.reg_tcn_num_channels[i - 1]
            out_channels = config.reg_tcn_num_channels[i]

            # Calculate padding to maintain sequence length
            padding = (config.reg_tcn_kernel_size - config.reg_tcn_stride) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.reg_tcn_kernel_size,
                    stride=config.reg_tcn_stride,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=config.reg_tcn_dropout,
                    activation_fn=activation_fn,
                )
            )

        self.tcn_network = nn.Sequential(*layers)

        # Final convolutional layer to map to desired output channels
        self.conv_output = nn.Conv1d(
            config.reg_tcn_num_channels[-1],
            config.output_dim * config.out_seq_len,
            kernel_size=1,
        )

        # Global average pooling will reduce the sequence length dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Linear layer for final output
        self.linear = nn.Linear(
            config.output_dim * config.out_seq_len,
            config.output_dim * config.out_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x has shape (batch_size, seq_len, n_embd)
        # Reshape to (batch_size, n_embd, seq_len) for conv1d
        x = x.transpose(1, 2)

        # Process through TCN network
        features = self.tcn_network(x)

        # Map to output channels
        x = self.conv_output(features)

        # Apply global average pooling to reduce sequence length
        x = self.pool(x)

        # Reshape to (batch_size, output_dim * out_seq_len)
        x = x.view(x.size(0), -1)

        # Apply linear layer for final output
        return self.linear(x)


@dataclass
class GPTConfig:
    in_seq_len: int = 5 * 5  # input sequence length
    out_seq_len: int = 11  # output sequence length
    input_dim: int = 1  # dimension of input features
    output_dim: int = 1  # dimension of output (target) features
    n_layer: int = 8  # number of transformer layers
    n_head: int = 4  # number of attention heads
    n_embd: int = 32  # embedding dimension
    dropout: float = 0.1  # dropout rate
    bias: bool = False  # use bias in linear layers
    is_causal: bool = True  # Whether to use causal masking in self-attention
    
    # RegressionTCN hyperparameters
    reg_tcn_kernel_size: int = 7  # Kernel size for TCN layers
    reg_tcn_num_channels: list[int] = None  # Number of channels in each TCN layer
    reg_tcn_dilation_base: int = 2  # Base for dilation
    reg_tcn_stride: int = 1  # Stride for TCN layers
    reg_tcn_activation: str = 'LeakyReLU'  # Activation function
    reg_tcn_dropout: float = 0.1  # Dropout rate for TCN

    def __post_init__(self):
        if self.reg_tcn_num_channels is None:
            self.reg_tcn_num_channels = [16, 32, 64, 64]  # Default channel configuration


class GPT(nn.Module):
    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        assert config.input_dim is not None
        assert config.in_seq_len is not None
        assert config.out_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Replace embedding with linear projection for continuous inputs
                wte=nn.Linear(config.input_dim, config.n_embd),
                # Keep positional embeddings for sequence structure
                wpe=nn.Embedding(config.in_seq_len, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # regression head
        # We use a TCN here, because a simple MLP or Linear layer would be enormous
        self.regression_head = RegressionTCN(config)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()

        embedding_params = sum(p.numel() for p in self.transformer.wte.parameters())
        position_params = sum(p.numel() for p in self.transformer.wpe.parameters())
        block_params = sum(p.numel() for p in self.transformer.h.parameters())
        regression_params = sum(p.numel() for p in self.regression_head.parameters())

        logger.info('Number of embedding params: %.3fM', embedding_params / 1e6)
        logger.info('Number of position params: %.3fM', position_params / 1e6)
        logger.info('Number of block params: %.3fM', block_params / 1e6)
        logger.info('Number of regression params: %.3fM', regression_params / 1e6)

        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        device = x.device
        x = x.squeeze(1)
        x = x.unsqueeze(-1)
        batch_size, seq_len, n_features = x.size()  # batch, sequence length, features
        assert seq_len <= self.config.in_seq_len, (
            f'Cannot forward sequence of length {seq_len}, '
            f'block size is only {self.config.in_seq_len}'
        )

        # Input sequence processing
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        # project inputs to embedding dimension (batch_size, seq_len, n_embd)
        tok_emb = self.transformer.wte(x)
        # position embeddings of shape (seq_len, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return self.regression_head(x)
