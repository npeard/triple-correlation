"""Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# source: https://github.com/karpathy/nanoGPT/blob/master/model.py


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
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
            print(
                'WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0'
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
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
            # print("using flash attention")
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
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RegressionCNN(nn.Module):
    """CNN-based regression head that takes data with n_embd channels of length in_seq_len
    and applies convolutional layers followed by average pooling to get the final output.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=config.n_embd,
            out_channels=config.n_embd * 2,
            kernel_size=3,
            padding=1,
        )

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=config.n_embd * 2,
            out_channels=config.n_embd * 2,
            kernel_size=3,
            padding=1,
        )

        # Final convolutional layer to reduce to desired output channels
        self.conv3 = nn.Conv1d(
            in_channels=config.n_embd * 2,
            out_channels=config.output_dim * config.out_seq_len,
            kernel_size=1,
        )

        # Activation function
        self.activation = nn.GELU()

        # Global average pooling will reduce the sequence length dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Linear layer for final output
        self.linear = nn.Linear(
            config.output_dim * config.out_seq_len,
            config.output_dim * config.out_seq_len,
        )

    def forward(self, x):
        # Input x has shape (batch_size, seq_len, n_embd)
        # Reshape to (batch_size, n_embd, seq_len) for conv1d
        x = x.transpose(1, 2)

        # Apply convolutional layers with activations
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        # Apply global average pooling to reduce sequence length
        x = self.pool(x)

        # Reshape to (batch_size, output_dim * out_seq_len)
        x = x.view(x.size(0), -1)

        # Apply linear layer for final output
        x = self.linear(x)

        return x


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


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
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
        # We use a CNN here, because a simple MLP or Linear layer would be enormous
        self.regression_head = RegressionCNN(config)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print('Total number of parameters: %.2fM' % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=False):
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

        print('Number of embedding params: %.3fM' % (embedding_params / 1e6,))
        print('Number of position params: %.3fM' % (position_params / 1e6,))
        print('Number of block params: %.3fM' % (block_params / 1e6,))
        print('Number of regression params: %.3fM' % (regression_params / 1e6,))

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        x = x.squeeze(1)
        x = x.unsqueeze(-1)
        batch_size, seq_len, n_features = x.size()  # batch, sequence length, features
        assert seq_len <= self.config.in_seq_len, (
            f'Cannot forward sequence of length {seq_len}, block size is only {self.config.in_seq_len}'
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

        predictions = self.regression_head(x)

        return predictions
