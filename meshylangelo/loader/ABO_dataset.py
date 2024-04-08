import os
import json
import random
import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

TEST_INDEX = 50

class ABODataset(Dataset):
    def __init__(
        self,
        data_root,
        mode="train"
    ):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        # filter
        self.load_entries(data_root, mode)
    
    def load_entries(self, data_root, mode="train"):
        if mode == "train":
            entry_list_file = os.path.join(data_root, "train_set.txt")
        elif mode == "test":
            entry_list_file = os.path.join(data_root, "test_set.txt")
        else:
            raise NotImplementedError(f"[Error] unrecognized mode: {mode}")
        
        self.entries = []
        with open(entry_list_file, "r") as f:
            name = f.readline().strip()
            while name:
                entry = os.path.join(data_root, name)
                if os.path.isdir(entry) and os.path.isfile(os.path.join(entry, "mesh", "latent.npz")):
                    if os.path.isdir(os.path.join(entry, "images")):
                        self.entries.append(entry)
                name = f.readline().strip()
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        entry = self.entries[index]
        # load latent
        latent_path = os.path.join(entry, "mesh", "latent.npz")
        latents = np.load(latent_path)['latent']
        # load image
        with open(os.path.join(entry, "transforms.json"), 'r') as fp:
            frames = json.load(fp=fp)["frames"]
        if self.mode == "train":
            frame = random.choice(frames)
        else:
            frame = frames[TEST_INDEX]
        image_path = os.path.join(entry, frame['file_path'])
        image = Image.open(image_path)
        background_color = np.random.uniform(low=0.0, high=255.0, size=3).astype(np.uint8) # give random background color
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, tuple(background_color))
            background.paste(image, image.split()[-1])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        image = np.transpose(image, (2, 0, 1))
        
        return {
            "latents": torch.from_numpy(latents).to(torch.float32).squeeze(0),
            "images": torch.from_numpy(image).to(torch.float32),
        }
        
if __name__ == "__main__":
    ds = ABODataset("/home/chuanyu/Desktop/ShapeInit/data")
    # print(len(ds))
    # for k in ds[0]:
    #     print(k, ds[0][k].size())
    std_array = []
    for i in range(len(ds)):
        std_array.append(ds[i]["latents"].std().item())
    
    std_array = np.array(std_array)
    print(std_array.mean(), std_array.min(), std_array.max(), std_array.std()) 
    

        