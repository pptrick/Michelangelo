# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from typing import Optional, List
import argparse
from functools import partial

from einops import repeat, rearrange
import numpy as np
from PIL import Image
import trimesh
import cv2

import torch
import pytorch_lightning as pl

from michelangelo.models.tsal.tsal_base import Latent2MeshOutput
from michelangelo.models.tsal.inference_utils import extract_geometry
from michelangelo.utils.misc import get_config_from_file, instantiate_from_config
from michelangelo.utils.visualizers.pythreejs_viewer import PyThreeJSViewer
from michelangelo.utils.visualizers import html_util
from michelangelo.utils.sampler import MeshSampler

def load_model(args):

    model_config = get_config_from_file(args.config_path)
    if hasattr(model_config, "model"):
        model_config = model_config.model

    model = instantiate_from_config(model_config, ckpt_path=args.ckpt_path)
    model = model.cuda()
    model = model.eval()

    return model

def load_surface(fp):
    
    with np.load(args.pointcloud_path) as input_pc:
        surface = input_pc['points']
        normal = input_pc['normals']
    
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], 4096, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

def prepare_image(args, number_samples=2):
    
    image = cv2.imread(f"{args.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_pt = torch.tensor(image).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    
    image_pt = repeat(image_pt, "c h w -> b c h w", b=number_samples)

    return image_pt

def save_output(args, mesh_outputs):
    
    os.makedirs(args.output_dir, exist_ok=True)
    for i, mesh in enumerate(mesh_outputs):
        mesh.mesh_f = mesh.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

        name = str(i) + "_out_mesh.obj"
        mesh_output.export(os.path.join(args.output_dir, name), include_normals=True)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')        

    return 0

def save_pcld(args, surface:torch.Tensor):
    os.makedirs(args.output_dir, exist_ok=True)
    points = surface[..., 0:3].detach().cpu().numpy().reshape(-1, 3)
    normals = surface[..., 3:6].detach().cpu().numpy().reshape(-1, 3)
    normals = (normals + 1.0) / 2.0
    with open(os.path.join(args.output_dir, "original_pcld.obj"), 'w') as f:
        N = len(points)
        for i in range(N):
            # write vertex, with normal color
            f.write(f"v {points[i][0]} {points[i][1]} {points[i][2]} {normals[i][0]} {normals[i][1]} {normals[i][2]} \n")

def reconstruction(args, model, bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), octree_depth=7, num_chunks=10000):

    if str(args.pointcloud_path).endswith(".npz"):
        surface = load_surface(args.pointcloud_path)
    else:
        mesh_sampler = MeshSampler(args.pointcloud_path)
        surface = mesh_sampler.sample(n_points=4096)
        surface = torch.FloatTensor(surface).unsqueeze(0).cuda()
        
    # encoding
    shape_embed, shape_latents = model.model.encode_shape_embed(surface, return_latents=True)    
    shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents)
    # decoding
    latents = model.model.shape_model.decode(shape_zq)
    geometric_func = partial(model.model.shape_model.query_geometry, latents=latents)
    
    # reconstruction
    mesh_v_f, has_surface = extract_geometry(
        geometric_func=geometric_func,
        device=surface.device,
        batch_size=surface.shape[0],
        bounds=bounds,
        octree_depth=octree_depth,
        num_chunks=num_chunks,
    )
    recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
    
    # save
    os.makedirs(args.output_dir, exist_ok=True)
    recon_mesh.export(os.path.join(args.output_dir, 'reconstruction.obj'))   
    
    save_pcld(args, surface) 
    
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {os.path.join(args.output_dir, "reconstruction.obj")}')
    print(f'-----------------------------------------------------------------------------')
    
    return 0

def image2mesh(args, model, guidance_scale=7.5, box_v=1.1, octree_depth=7):

    sample_inputs = {
        "image": prepare_image(args)
    }
    
    mesh_outputs = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth,
    )[0]
    
    save_output(args, mesh_outputs)
    
    return 0

def text2mesh(args, model, num_samples=2, guidance_scale=7.5, box_v=1.1, octree_depth=7):

    sample_inputs = {
        "text": [args.text] * num_samples
    }
    mesh_outputs = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth,
    )[0]
    
    save_output(args, mesh_outputs)
    
    return 0

task_dick = {
    'reconstruction': reconstruction,
    'image2mesh': image2mesh,
    'text2mesh': text2mesh,
}

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['reconstruction', 'image2mesh', 'text2mesh'], required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, default='./example_data/surface.npz', help='Path to the input point cloud')
    parser.add_argument("--image_path", type=str, help='Path to the input image')
    parser.add_argument("--text", type=str, help='Input text within a format: A 3D model of motorcar; Porsche 911.')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running {args.task}')
    args.output_dir = os.path.join(args.output_dir, args.task)
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    task_dick[args.task](args, load_model(args))