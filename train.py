import os
import cv2
import torch
from einops import repeat, rearrange

import trimesh

from torch.utils.data import DataLoader

from meshylangelo.diffusion.trainer import Trainer
from meshylangelo.loader.ABO_dataset import ABODataset

# TODO:
# Two things to be checked:
# 1. latents std is far larger than 1.0, need to be checked
# 2. what's the latent shape for the denoiser? [257, 64] or [256, 64]? The VAE can take both of them.

dataset = ABODataset("/home/chuanyu/Desktop/ShapeInit/data", mode="test")
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

trainer = Trainer(
    device="cuda",
    lr=1e-4,
    n_epoch=100,
    # denoiser_ckpt_path= "./meshylangelo/diffusion/checkpoints/denoiser-ASLDM-256.ckpt"
    denoiser_ckpt_path= "./exp/training/checkpoints/latest_denoiser.ckpt"
)

# trainer.train(data_loader)

# def prepare_image(image_path, number_samples=1):
    
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     image_pt = torch.tensor(image).float()
#     image_pt = image_pt / 255 * 2 - 1
#     image_pt = rearrange(image_pt, "h w c -> c h w")
    
#     image_pt = repeat(image_pt, "c h w -> b c h w", b=number_samples)

#     return image_pt

# data = {
#     "images": prepare_image("./example_data/image/pikachu.png")
# }

data = dataset[1]
for k in data:
    data[k] = data[k].unsqueeze(0)
    
# print(data["images"].size())
# exit()

outputs = trainer.sample(data, sample_times=1, guidance_scale=7.5, steps=50)

for i, mesh in enumerate(outputs):
    mesh.mesh_f = mesh.mesh_f[:, ::-1]
    mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

    name = str(i) + "_out_mesh.obj"
    mesh_output.export(name, include_normals=True)






