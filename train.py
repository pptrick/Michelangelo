import os
import trimesh

from torch.utils.data import DataLoader

from meshylangelo.diffusion.trainer import Trainer
from meshylangelo.loader.ABO_dataset import ABODataset

dataset = ABODataset("/home/chuanyu/Desktop/ShapeInit/data")
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

trainer = Trainer(
    device="cuda",
    lr=1e-4,
    n_epoch=100,
    denoiser_ckpt_path= "./exp/training/checkpoints/latest_denoiser.ckpt" # "./meshylangelo/diffusion/checkpoints/denoiser-ASLDM-256.ckpt"
)

trainer.train(data_loader)

# data = dataset[1]
# for k in data:
#     data[k] = data[k].unsqueeze(0)

# outputs = trainer.sample(data, sample_times=1, guidance_scale=7.5, steps=50)

# for i, mesh in enumerate(outputs):
#     mesh.mesh_f = mesh.mesh_f[:, ::-1]
#     mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

#     name = str(i) + "_out_mesh.obj"
#     mesh_output.export(name, include_normals=True)






