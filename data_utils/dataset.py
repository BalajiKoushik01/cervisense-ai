import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from glob import glob
from collections import Counter
import numpy as np

def _load_image(path):
    # Ensure RGB, handles PNG and JPG
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

class CervicalSSLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = glob(os.path.join(data_dir, "*.*"))
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = _load_image(img_path)
        if image is None: # return random if failed
             return self.__getitem__(random.randint(0, len(self)-1))
             
        # Convert PIL to numpy for albumentations
        image = np.array(image)
        
        # Dual augmentation for MoCo
        if self.transform:
            x1 = self.transform(image=image)["image"]
            x2 = self.transform(image=image)["image"]
        else:
            x1, x2 = image, image
            
        return x1, x2

class CervicalSupervisedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = _load_image(img_path)
        if image is None:
             return self.__getitem__(random.randint(0, len(self)-1))
             
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
        
    def get_weighted_sampler(self):
        labels = [s[1] for s in self.samples]
        class_counts = Counter(labels)
        class_weights = {k: 1.0 / count for k, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.samples), replacement=True)
        return sampler

class DualDomainDataset(Dataset):
    def __init__(self, colpo_dir, histo_dir, transform=None):
        self.transform = transform
        self.classes = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
        
        # Organize by class
        self.colpo_samples = {cls: [] for cls in self.classes}
        self.histo_samples = {cls: [] for cls in self.classes}
        
        for cls_name in self.classes:
            c_dir = os.path.join(colpo_dir, cls_name)
            if os.path.exists(c_dir):
                self.colpo_samples[cls_name] = glob(os.path.join(c_dir, "*.*"))
                
            h_dir = os.path.join(histo_dir, cls_name)
            if os.path.exists(h_dir):
                self.histo_samples[cls_name] = glob(os.path.join(h_dir, "*.*"))
                
        self.num_samples = sum(len(v) for v in self.colpo_samples.values())
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Pick a random class that has both (or fallback)
        cls_name = random.choice(self.classes)
        while not (self.colpo_samples[cls_name] and self.histo_samples[cls_name]):
            cls_name = random.choice(self.classes)
            
        c_path = random.choice(self.colpo_samples[cls_name])
        h_path = random.choice(self.histo_samples[cls_name])
        
        c_img = _load_image(c_path)
        h_img = _load_image(h_path)
        
        if c_img is None or h_img is None:
            return self.__getitem__(idx) # Retry on failure
            
        c_img = np.array(c_img)
        h_img = np.array(h_img)
        
        if self.transform:
            c_img = self.transform(image=c_img)["image"]
            h_img = self.transform(image=h_img)["image"]
            
        label = self.classes.index(cls_name)
        return c_img, h_img, label
