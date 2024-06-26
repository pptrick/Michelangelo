import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
from PIL import Image
import trimesh
from einops import repeat, rearrange
import numpy as np

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
        mode:str="train",
        denoiser_ckpt_path:Optional[str] = None,
        vae_ckpt_path:Optional[str] = "./meshylangelo/vae/checkpoints/shapevae-256.ckpt",
        lr:float=1e-4,
        n_epoch:int=100
    ):
        self.n_epoch = n_epoch
        self.out_dir = out_dir
        self.mode = mode
        self.trainer = Trainer(
            outdir=out_dir,
            device=device,
            mode=mode,
            denoiser_ckpt_path=denoiser_ckpt_path,
            vae_ckpt_path=vae_ckpt_path,
            lr=lr,
            n_epoch=n_epoch
        )
        
        self.dataset = None
        self.data_mode = "train"
        
    def load_dataset(self, data_root, name="ABO", batch_size=4):
        if name == "ABO":
            self.dataset = ABODataset(data_root=data_root, mode=self.mode)
        else:
            raise NotImplementedError(f"[Error] dataset class {name} is not implemented!")
        
        if self.mode == "train":
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
            self.data_mode = "train"
        elif self.mode == "test":
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())
            self.data_mode = "test"
        else:
            raise NotImplementedError(f"[Error] unrecognized dataset mode: {self.mode}")
        
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
        
        # save image
        image = data["images"][0].cpu()
        image = rearrange(image, "c h w -> h w c")
        image = ((image + 1) * 255.0 / 2.0).numpy().astype(np.uint8)
        Image.fromarray(image).save(os.path.join(outdir, "input.png"))
        
                
def parse_args():
    parser = argparse.ArgumentParser(prog="train/test meshylangelo diffusion model")
    parser.add_argument('--task', type=str, default="training", choices=["training", "testing"], help='type of task')
    
    parser.add_argument('--outdir', type=str, default="exp", help='output directory')
    parser.add_argument('--denoiser_ckpt', type=str, default=None, help='path to pretrained denoiser checkpoint')
    parser.add_argument('--vae_ckpt', type=str, default="./meshylangelo/vae/checkpoints/shapevae-256.ckpt", help='path to vae checkpoint')
    
    # dataset
    parser.add_argument('--data_root', type=str, default="/mnt/storage/ABO-nerf", help='path to data root directory')
    parser.add_argument('--data_name', type=str, default="ABO", choices=["ABO"], help='name of the dataset')
    
    # training
    parser.add_argument('--n_epoch', type=int, default=500, help='number of training epochs, only valid when training')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size, only valid when training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, only valid when training')
    
    # testing
    parser.add_argument('--test_data_index', type=int, default=1, help='testing index of dataset, only valid when testing')
    parser.add_argument('--img_path', type=str, default=None, help='path to input image, only valid when testing')
    parser.add_argument('--num_samples', type=int, default=2, help='number of testing samples output, only valid when testing')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='guidance scale, only valid when testing')
    parser.add_argument('--steps', type=int, default=50, help='sample steps, only valid when testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task = args.task
    
    if task == "training":
        train_manager = TrainingManager(
            out_dir=args.outdir,
            mode="train",
            vae_ckpt_path=args.vae_ckpt,
            denoiser_ckpt_path=args.denoiser_ckpt,
            lr=args.lr,
            n_epoch=args.n_epoch
        )
        train_manager.load_dataset(
            data_root=args.data_root,
            name=args.data_name,
            batch_size=args.batch_size
        )
        train_manager.train()
    
    elif task == "testing":
        train_manager = TrainingManager(
            out_dir=args.outdir,
            mode="test",
            vae_ckpt_path=args.vae_ckpt,
            denoiser_ckpt_path=args.denoiser_ckpt,
            lr=args.lr,
            n_epoch=args.n_epoch
        )
        train_manager.load_dataset(
            data_root=args.data_root,
            name=args.data_name,
        )
        train_manager.test(
            img_path=args.img_path,
            data_index=args.test_data_index,
            num_samples=args.num_samples,
            guidance_scale=args.guidance_scale,
            steps=args.steps
        )






