# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple, Dict, Optional, Union

from transformers import CLIPModel

from meshylangelo.vae.shape_module import AlignedShapeLatentPerceiver, ShapeAsLatentPerceiver
from meshylangelo.vae.utils import Latent2MeshOutput, extract_geometry

class SITA_VAE(nn.Module):
    def __init__(self, *,
                 clip_model_version: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        
        self.clip_model: CLIPModel = CLIPModel.from_pretrained(clip_model_version)
        
        # TODO: hard coding shape model
        self.shape_model: AlignedShapeLatentPerceiver = AlignedShapeLatentPerceiver(
            device=None, dtype=None,
            num_latents=256,
            embed_dim=64,
            point_feats=3,
            num_freqs=8,
            include_pi=False,
            heads=12,
            width=768,
            num_encoder_layers=8,
            num_decoder_layers=16,
            use_ln_post=True,
            init_scale=0.25,
            qkv_bias=False,
            use_checkpoint=True
        )
        
        # self.shape_model: ShapeAsLatentPerceiver = ShapeAsLatentPerceiver(
        #     device=None, dtype=None,
        #     num_latents=257,
        #     embed_dim=64,
        #     point_feats=3,
        #     num_freqs=8,
        #     include_pi=False,
        #     heads=12,
        #     width=768,
        #     num_encoder_layers=8,
        #     num_decoder_layers=16,
        #     use_ln_post=True,
        #     init_scale=0.25,
        #     qkv_bias=False,
        #     use_checkpoint=True
        # )
        
        # shape projection and init
        self.shape_projection = nn.Parameter(torch.empty(self.shape_model.width, self.clip_model.projection_dim))
        nn.init.normal_(self.shape_projection, std=self.clip_model.projection_dim ** -0.5)
        
    @property
    def latent_shape(self):
        return self.shape_model.latent_shape
        
    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq
    
    def decode(self, z_q):
        
        latents = self.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        return latents
    
    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000,
                    disable: bool = False) -> List[Latent2MeshOutput]:

        """

        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = []

        geometric_func = partial(self.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=disable
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs
        
        
    
