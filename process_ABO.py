import os
import tqdm

import torch
import numpy as np

from meshylangelo.utils.sampler import MeshSampler

class ABOLoader:
    def __init__(self, dataroot):
        self.models_folder = os.path.join(dataroot, "3dmodels", "original")
        self.categories = sorted(os.listdir(self.models_folder))
        self.models = []
        for c in self.categories:
            self.models += [os.path.join(self.models_folder, c, f) for f in sorted(os.listdir(os.path.join(self.models_folder, c))) if f.endswith(".glb")]
        
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, index):
        return self.models[index]

@torch.no_grad()
def process(src_dir, tar_dir, vae_model):
    loader = ABOLoader(src_dir)
    
    for i in tqdm.tqdm(range(len(loader))):
        out_dir = os.path.join(tar_dir, f"{'%04d' % i}")
        assert os.path.exists(out_dir)
        out_dir = os.path.join(out_dir, "mesh")
        os.makedirs(out_dir, exist_ok=True)
        
        sampler = MeshSampler(mesh_path=loader[i])
        surface = sampler.sample(n_points=10000)
        surface = torch.FloatTensor(surface).unsqueeze(0).cuda()
        
        # shape_zq = model.encode(surface=surface, sample_posterior=True) # [1, 256, 64]
        pc = surface[..., 0:3]
        feats = surface[..., 3:6]
        latents, _, _ = model.shape_model.encode(pc, feats) # SITA VAE with ShapeAsLatentPerceiver
        np.savez_compressed(os.path.join(out_dir, "latent.npz"), latent=latents.cpu().numpy())
        
    
if __name__ == "__main__":
    from meshylangelo.vae.sita_vae import SITA_VAE
    model = SITA_VAE()
    model.load_state_dict(torch.load("./meshylangelo/vae/checkpoints/shapevae-256.ckpt", map_location="cpu"), strict=False)
    model = model.cuda().eval()
    
    process(
        src_dir="/home/chuanyu/Desktop/Dataset/ABO",
        tar_dir="/home/chuanyu/Desktop/ShapeInit/data",
        vae_model=model
    )