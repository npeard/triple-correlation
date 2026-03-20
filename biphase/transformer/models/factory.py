#!/usr/bin/env python

"""Model factory for creating nn.Module instances."""

import inspect

from torch import nn

from biphase.transformer.models.gpt import GPT, GPTConfig


def _create_gpt_config(model_hparams: dict) -> GPTConfig:
    """Create GPTConfig from model hyperparameters.

    Computes in_seq_len and out_seq_len from num_pix, then filters
    remaining params to match GPTConfig fields.
    """
    num_pix = model_hparams['num_pix']
    if isinstance(num_pix, str):
        num_pix = eval(num_pix)

    if isinstance(num_pix, tuple):
        num_pix = num_pix[0]
        Phi_dim = num_pix // 2 + 1
        Phi_dim -= 1
        in_seq_len = Phi_dim**4
        out_seq_len = num_pix**2
    elif isinstance(num_pix, int):
        Phi_dim = num_pix // 2 + 1
        Phi_dim -= 1
        in_seq_len = Phi_dim**2
        out_seq_len = num_pix
    else:
        raise TypeError(f'Unsupported type for num_pix: {type(num_pix)}')

    gpt_config_fields = set(inspect.signature(GPTConfig).parameters.keys())
    filtered_params = {k: v for k, v in model_hparams.items() if k in gpt_config_fields}

    return GPTConfig(in_seq_len=in_seq_len, out_seq_len=out_seq_len, **filtered_params)


def create_model(model_type: str, model_hparams: dict) -> nn.Module:
    """Factory method to create nn.Module models.

    Args:
        model_type: Type of model to create (e.g. 'GPT')
        model_hparams: Model hyperparameters dict

    Returns:
        nn.Module: The created model
    """
    if model_type == 'GPT':
        config = _create_gpt_config(model_hparams)
        return GPT(config)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
