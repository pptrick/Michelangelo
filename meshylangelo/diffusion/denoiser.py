# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.embeddings import Timesteps
import math

from meshylangelo.modules.transformer_blocks import MLP
from meshylangelo.modules.diffusion_transformer import UNetDiffusionTransformer, DiT

def position_encoding(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class AttnUnetDenoiser(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 input_channels: int,
                 output_channels: int,
                 n_ctx: int,
                 width: int,
                 layers: int,
                 heads: int,
                 context_dim: int,
                 context_ln: bool = True,
                 skip_ln: bool = False,
                 init_scale: float = 0.25,
                 flip_sin_to_cos: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.backbone = UNetDiffusionTransformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            skip_ln=skip_ln,
            init_scale=init_scale,
            use_checkpoint=use_checkpoint
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)

        # timestep embedding
        self.time_embed = Timesteps(width, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=0)
        self.time_proj = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale
        )

        if context_ln:
            self.context_embed = nn.Sequential(
                nn.LayerNorm(context_dim, device=device, dtype=dtype),
                nn.Linear(context_dim, width, device=device, dtype=dtype),
            )
        else:
            self.context_embed = nn.Linear(context_dim, width, device=device, dtype=dtype)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        _, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_proj(self.time_embed(timestep)).unsqueeze(dim=1)

        # 2. conditions projector
        context = self.context_embed(context)

        # 3. denoiser
        x = self.input_proj(model_input)
        # pe = position_encoding(length=x.size(1), d_model=x.size(2))
        # x += pe.to(x.dtype).to(x.device).unsqueeze(0)
        x = torch.cat([t_emb, context, x], dim=1)
        
        x = self.backbone(x)
        x = self.ln_post(x)
        x = x[:, -n_data:]
        sample = self.output_proj(x)

        return sample
    
    
# TODO: check DiTDenoiser (use DiT model)
class DiTDenoiser(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 input_channels: int,
                 output_channels: int,
                 n_ctx: int,
                 width: int,
                 layers: int,
                 heads: int,
                 context_dim: int,
                 context_ln: bool = True,
                 skip_ln: bool = False,
                 init_scale: float = 0.25,
                 flip_sin_to_cos: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        init_scale = init_scale * math.sqrt(1.0 / width)
        
        self.backbone = DiT(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            context_dim=context_dim,
            init_scale=init_scale,
            use_checkpoint=use_checkpoint
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)

        # timestep embedding
        self.time_embed = Timesteps(width, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=0)
        self.time_proj = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale
        )

        if context_ln:
            self.context_embed = nn.Sequential(
                nn.LayerNorm(context_dim, device=device, dtype=dtype),
                nn.Linear(context_dim, width, device=device, dtype=dtype),
            )
        else:
            self.context_embed = nn.Linear(context_dim, width, device=device, dtype=dtype)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        _, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_proj(self.time_embed(timestep)).unsqueeze(dim=1)

        # 2. conditions projector
        # context = self.context_embed(context)

        # 3. denoiser
        x = self.input_proj(model_input)
        x = self.backbone(x, t_emb, context)
        
        x = self.ln_post(x)
        x = x[:, -n_data:]
        sample = self.output_proj(x)

        return sample