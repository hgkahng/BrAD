
import os
import json
import shutil

import gdown

import torch
import numpy as np
from PIL import Image


PACS_LABELS = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
PACS_DOMAINS = ('photo', 'art_painting', 'cartoon', 'sketch')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

class PACSDataset(object):
    def __init__(self, root, transform=None, target_transform=None,
                 test=False, return_path=False):
        self.torch_dataset = False
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.return_path = return_path
        
        self.imgs = []
        self.labels = []

    def __getitem__(self, index):
        
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def download(root: str) -> None:
        """Download & extract PACS dataset."""
        
        url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
        dst = "PACS.zip"

        os.makedirs(root, exist_ok=True)
        
        # download
        _dst = os.path.join(root, dst)
        if not os.path.exists(_dst):
            gdown.download(url, _dst, quiet=False)
        
        # extract
        from zipfile import ZipFile
        zf = ZipFile(_dst, "r")
        zf.extractall(os.path.dirname(_dst))
        zf.close()
        
        # change folder hierarchy
        inner_dir = os.path.join(root, 'kfold')
        for subdir in os.listdir(inner_dir):
            subdir = os.path.join(inner_dir, subdir)
            shutil.move(subdir, root)

        os.removedirs(inner_dir)
