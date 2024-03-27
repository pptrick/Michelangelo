import os
import argparse

import torch
from functools import partial

import trimesh
import numpy as np

from meshylangelo.vae.sita_vae import SITA_VAE
from meshylangelo.utils.sampler import MeshSampler

def load_surface(pointcloud_path):
    
    with np.load(pointcloud_path) as input_pc:
        surface = input_pc['points']
        normal = input_pc['normals']
    
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], 4096, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

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


def reconstruction(args, model:SITA_VAE, bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), octree_depth=7, num_chunks=10000):

    if str(args.pointcloud_path).endswith(".npz"):
        surface = load_surface(args.pointcloud_path)
    else:
        mesh_sampler = MeshSampler(args.pointcloud_path)
        surface = mesh_sampler.sample(n_points=4096)
        surface = torch.FloatTensor(surface).unsqueeze(0).cuda()
        
    # encoding
    shape_zq = model.encode(surface=surface, sample_posterior=True)
    # decoding
    latents = model.decode(shape_zq)
    
    # reconstruction
    meshes = model.latent2mesh(
        latents=latents,
        bounds=bounds,
        octree_depth=octree_depth,
        num_chunks=num_chunks
    )
    recon_mesh = trimesh.Trimesh(meshes[0].mesh_v, meshes[0].mesh_f)
    
     # save
    os.makedirs(args.output_dir, exist_ok=True)
    recon_mesh.export(os.path.join(args.output_dir, 'reconstruction.obj'))   
    
    save_pcld(args, surface) 
    
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {os.path.join(args.output_dir, "reconstruction.obj")}')
    print(f'-----------------------------------------------------------------------------')
    
    return 0


def filter_ckpt(ckpt_path):
    from collections import OrderedDict
    old_state_dict = torch.load(ckpt_path)['state_dict']
    new_state_dict = OrderedDict()
    for k in old_state_dict:
        if k.startswith("model.clip_model"):
            continue
        elif k.startswith("model.shape_model"):
            new_state_dict[k.replace("model.shape_model", "shape_model")] = old_state_dict[k]
        else:
            new_state_dict[k.replace("model.", "")] = old_state_dict[k]
    torch.save(new_state_dict, "shapevae-256.ckpt")

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reconstruction", type=str, choices=['reconstruction', 'image2mesh', 'text2mesh'])
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, default='./example_data/surface.npz', help='Path to the input point cloud')
    parser.add_argument("--output_dir", type=str, default='./output_test')
    args = parser.parse_args()

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running {args.task}')
    args.output_dir = os.path.join(args.output_dir, args.task)
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    model = SITA_VAE()
    model.load_state_dict(torch.load("./meshylangelo/vae/checkpoints/shapevae-256.ckpt", map_location="cpu"), strict=False)
    
    model = model.cuda().eval()
    reconstruction(args=args, model=model)