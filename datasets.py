import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.data_files = sorted(glob.glob(os.path.join(root, "data") + "/*.*"))
        self.label_files = sorted(glob.glob(os.path.join(root, "label") + "/*.*"))

        if mode == "train":
            self.data_files = self.data_files[:]
            self.label_files = self.label_files[:]
        else:
            self.data_files = self.data_files[:]
            self.label_files = self.label_files[:]

    def __getitem__(self, index):

        data_img = Image.open(self.data_files[index % len(self.data_files)])
        label_img = Image.open(self.label_files[index % len(self.label_files)])

        img_A = self.transform(data_img)
        img_B = self.transform(label_img)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.data_files)
