import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import trimesh
from einops import repeat, rearrange

import torch
from torch.utils.data import DataLoader

from meshylangelo.diffusion.trainer import Trainer
from meshylangelo.loader.ABO_dataset import ABODataset
    

def prepare_image(image_path, number_samples=1):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_pt = torch.tensor(image).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    
    image_pt = repeat(image_pt, "c h w -> b c h w", b=number_samples)

    return image_pt
    
class TrainingManager:
    def __init__(
        self,
        out_dir:str|Path = "exp",
        device:str|torch.device = "cuda",
        denoiser_ckpt_path:Optional[str] = None,
        vae_ckpt_path:Optional[str] = "./meshylangelo/vae/checkpoints/shapevae-256.ckpt",
        lr:float=1e-4,
        n_epoch:int=100
    ):
        self.n_epoch = n_epoch
        self.out_dir = out_dir
        self.trainer = Trainer(
            outdir=out_dir,
            device=device,
            denoiser_ckpt_path=denoiser_ckpt_path,
            vae_ckpt_path=vae_ckpt_path,
            lr=lr,
            n_epoch=n_epoch
        )
        
        self.dataset = None
        self.data_mode = "train"
        
    def load_dataset(self, data_root, name="ABO", mode="train", batch_size=4):
        if name == "ABO":
            self.dataset = ABODataset(data_root=data_root, mode=mode)
        else:
            raise NotImplementedError(f"[Error] dataset class {name} is not implemented!")
        
        if mode == "train":
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
            self.data_mode = "train"
        elif mode == "test":
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())
            self.data_mode = "test"
        else:
            raise NotImplementedError(f"[Error] unrecognized dataset mode: {mode}")
        
    def train(self):
        if self.dataset is None:
            print(f"[Error] dataset not initialized, call load_dataset() first!")
            exit()
        elif self.data_mode == "test":
            print(f"[Warning] current data mode is test, while you are training")
            
        print(f"[INFO] start training, epoch num: {self.n_epoch}...")
        self.trainer.train(self.dataloader)
        
    def test(
        self, 
        test_name:Optional[str]=None,
        img_path:Optional[str|Path]=None, 
        data_index:int=1, 
        num_samples:int=1,
        guidance_scale:float=7.5,
        steps:int=50,
        render_video:bool=True
    ):
        if test_name is None:
            outdir = os.path.join(self.out_dir, "testing", datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        else:
            outdir = os.path.join(self.out_dir, "testing", test_name)
        os.makedirs(outdir, exist_ok=True)
        
        if img_path is None:
            print(f"[INFO] no image input, use image from dataset...")
            if self.dataset is None:
                print(f"[Error] dataset not initialized, call load_dataset() first!")
                exit()
            elif self.data_mode == "train":
                print(f"[Warning] current data mode is train, while you are testing, use load_dataset() to update!")
            data = self.dataset[data_index]
            data["images"] = data["images"].unsqueeze(0).repeat(num_samples, 1, 1, 1)
        else:
            print(f"[INFO] loading image from {img_path}...")
            data = {
                "images": prepare_image(img_path, number_samples=num_samples)
            }
            
        outputs = self.trainer.sample(
            data, 
            sample_times=1, 
            guidance_scale=guidance_scale, 
            steps=steps)
        
        for i, mesh in enumerate(outputs):
            mesh.mesh_f = mesh.mesh_f[:, ::-1]
            mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

            name = str(i) + "_out_mesh.obj"
            mesh_path = os.path.join(outdir, name)
            mesh_output.export(mesh_path, include_normals=True)
            
        if render_video:
            print("[INFO] render video...")
            from libmeshy import render_turntable_video_for_file
            for i in range(len(outputs)):
                mesh_path = os.path.join(outdir, str(i) + "_out_mesh.obj")
                render_turntable_video_for_file(
                    fp=mesh_path, 
                    extrat_images=True, 
                    watermark=False,
                    out_path=outdir, 
                    num_images=180, 
                    image_width=512, 
                    image_height=512
                )
                # clear cache
                for f in os.listdir(outdir):
                    if f.endswith(".webp") or f=="preview.png":
                        os.remove(os.path.join(outdir, f))
                shutil.rmtree(os.path.join(outdir, "assimp_gltf_out"))
                os.rename(os.path.join(outdir, "output.mp4"), os.path.join(outdir, str(i) + "_render.mp4"))
                
            
if __name__ == "__main__":
    train_manager = TrainingManager(
        out_dir="exp_ABO",
        # denoiser_ckpt_path= "./meshylangelo/diffusion/checkpoints/denoiser-ASLDM-256.ckpt",
        denoiser_ckpt_path= "./exp_ABO/training/checkpoints/latest_denoiser.ckpt",
        lr=1e-4,
        n_epoch=500
    )
    
    DATA_ROOT = "/home/chuanyu/Desktop/ShapeInit/data"
    
    train_manager.load_dataset(
        data_root=DATA_ROOT,
        name="ABO",
        mode="train",
        batch_size=4
    )
    train_manager.train()
    
    train_manager.load_dataset(
        data_root=DATA_ROOT,
        name="ABO",
        mode="test"
    )
    train_manager.test(
        img_path=None,
        data_index=1,
        num_samples=2,
        guidance_scale=7.5,
        steps=50
    )






