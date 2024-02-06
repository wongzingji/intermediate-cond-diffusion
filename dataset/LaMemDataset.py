"""from"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image

class LaMemEvalDataset(Dataset):

    def __init__(self, image_dir, csv_file=None, transform=None):
        self.score_list = None
        if csv_file is not None:
            self.score_list = pd.read_csv(csv_file, delim_whitespace=True, header=None)
        self.image_dir = image_dir
        self.filenames = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        if self.score_list is not None:
            return len(self.score_list)
        else:
            return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.score_list is not None:
            img_name = self.score_list.iloc[idx, 0]
            img_path = os.path.join(self.image_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")

            mem_score = self.score_list.iloc[idx, 1]
            target = float(mem_score)
            target = torch.tensor(target)
        else:
            img_name = self.filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")
            target = np.nan
        if self.transform:
            image = self.transform(image)
        
        return image, {'name': img_name, 'target': target}
