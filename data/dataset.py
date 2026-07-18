import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CropDataset(Dataset):
    def __init__(self, manifest_path, split_path, role, input_size, methods=None):
        manifest = pd.read_parquet(manifest_path)
        split = pd.read_csv(split_path)
        ids = split[split["role"] == role]["crop_id"]
        df = manifest[manifest["crop_id"].isin(ids)]
        if methods is not None:
            df = df[df["method"].isin(methods)]
        self.df = df.reset_index(drop=True)
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.load_image(row["path"])
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(row["label"]), row["method"], row["clip_id"]

    def load_image(self, path):
        if str(path).endswith(".npy"):
            return np.load(path)
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
