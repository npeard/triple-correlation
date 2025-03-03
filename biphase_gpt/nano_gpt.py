"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

# source: https://github.com/karpathy/nanoGPT/blob/master/model.py

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # print("k std: ", torch.std(k))
        # print("q std: ", torch.std(q))
        # print("v std: ", torch.std(v))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # print("using flash attention")
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # print("using manual attention")
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class ReductionMLP(nn.Module):
    """MLP for sequence length reduction, replacing the simple linear layer."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.block_size, 4 * config.output_block_size, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.output_block_size, config.output_block_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class RegressionMLP(nn.Module):
    """MLP for regression head, replacing the simple linear layer."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.output_dim, bias=config.bias)
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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    in_seq_len: int = 5*5      # input sequence length
    out_seq_len: int = 11      # output sequence length
    input_dim: int = 1         # dimension of input features
    output_dim: int = 1        # dimension of output (target) features
    n_layer: int = 8           # number of transformer layers
    n_head: int = 4            # number of attention heads
    n_embd: int = 32           # embedding dimension
    dropout: float = 0.1       # dropout rate
    bias: bool = False         # use bias in linear layers

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.input_dim is not None
        assert config.in_seq_len is not None
        assert config.out_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Replace embedding with linear projection for continuous inputs
            wte = nn.Linear(config.input_dim, config.n_embd),
            # Keep positional embeddings for sequence structure
            wpe = nn.Embedding(config.in_seq_len, config.n_embd),
            # Add output position embeddings
            wpe_out = nn.Embedding(config.out_seq_len, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Add sequence length reduction layer
        # Option 1: Simple linear reduction (current)
        self.seq_reduction = nn.Linear(config.in_seq_len, config.out_seq_len)
        # Option 2: MLP reduction (uncomment to use)
        # self.seq_reduction = ReductionMLP(config)

        # Replace language model head with regression head
        # Option 1: Simple linear regression (current)
        self.regression_head = nn.Linear(config.n_embd, config.output_dim)
        # Option 2: MLP regression (uncomment to use)
        # self.regression_head = RegressionMLP(config)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
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
        b, t, f = x.size()  # batch, sequence length, features
        assert t <= self.config.in_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.in_seq_len}"
        
        # Input sequence processing
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(x)  # project inputs to embedding dimension (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Reduce sequence length
        x = x.transpose(1, 2)  # (b, n_embd, t)
        x = self.seq_reduction(x)  # (b, n_embd, out_seq_len)
        x = x.transpose(1, 2)  # (b, out_seq_len, n_embd)
        # TODO: try removing output positional embeddings, and one of the reduction steps.
        # Add output positional embeddings
        out_pos = torch.arange(0, self.config.out_seq_len, dtype=torch.long, device=device)
        out_pos_emb = self.transformer.wpe_out(out_pos)
        x = x + out_pos_emb
        
        # Output continuous predictions
        predictions = self.regression_head(x)

        # implement antisymmetry of output sequence
        predictions = torch.cat([-1*predictions.flip(1), predictions[:, 1:]], dim=1)
        
        return predictions.squeeze(2)

    def get_attention_metrics(self):
        """
        Calculate the standard deviation of attention weights across all heads and layers.
        This helps monitor if the attention patterns are becoming too uniform (low std) 
        or maintaining meaningful distinctions (higher std).
        """
        attention_stds = []
        for block in self.transformer.h:
            # For each attention block, get the attention weights
            # Shape: [batch_size, n_head, sequence_length, sequence_length]
            with torch.no_grad():
                if block.attn.flash:
                    # For flash attention, we need to compute attention weights explicitly
                    q, k, v = block.attn.c_attn(torch.ones(1, self.config.in_seq_len, self.config.n_embd).to(next(self.parameters()).device)).split(self.config.n_embd, dim=-1)
                    q = q.view(1, self.config.in_seq_len, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(1, 2)
                    k = k.view(1, self.config.in_seq_len, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(1, 2)
                    v = v.view(1, self.config.in_seq_len, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(1, 2)
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    # print("att abs mean: ", torch.abs(att).mean())
                    att_abs_mean = torch.abs(att).mean().item()
                    att_std = torch.std(att, dim=-1).mean().item()
                    q_std = torch.std(q, dim=-1).mean().item()
                    k_std = torch.std(k, dim=-1).mean().item()
                    v_std = torch.std(v, dim=-1).mean().item()
                    att = F.softmax(att, dim=-1)
                    # print("att uniformity: ", torch.abs(att - 1/att.size(-1)).mean())
                    # print("att sparsity: ", (att.max(dim=-1).values > -.9).float().mean())
                    uniformity = torch.abs(att - 1/att.size(-1)).mean().item()
                    sparsity = (att.max(dim=-1).values > -.9).float().mean().item()
                else:
                    # For regular attention, we can access the attention weights directly
                    att = block.attn.att if hasattr(block.attn, 'att') else None
                
                if att is not None:
                    # Calculate std across the attention dimension
                    std = torch.std(att, dim=-1).mean()  # Mean across heads and sequence positions
                    attention_stds.append(std.item())
        
        return att_abs_mean, att_std, q_std, k_std, v_std, uniformity, sparsity 

    def get_position_encoding_std(self):
        """
        Calculate the standard deviation of position encodings.
        This helps monitor if the position encodings maintain meaningful positional information.
        """
        with torch.no_grad():
            # Get position embeddings for input sequence
            pos = torch.arange(0, self.config.in_seq_len, dtype=torch.long, 
                             device=next(self.parameters()).device)
            pos_emb = self.transformer.wpe(pos)  # [block_size, n_embd]
            
            # Calculate std across the embedding dimension for each position
            pos_std = torch.std(pos_emb, dim=-1).mean().item()
            
            # Get position embeddings for output sequence
            out_pos = torch.arange(0, self.config.output_block_size, dtype=torch.long, 
                                 device=next(self.parameters()).device)
            out_pos_emb = self.transformer.wpe_out(out_pos)  # [output_block_size, n_embd]
            
            # Calculate std across the embedding dimension for each output position
            out_pos_std = torch.std(out_pos_emb, dim=-1).mean().item()
            
            return (pos_std + out_pos_std) / 2.0  # Average of input and output position encoding stds