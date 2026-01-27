from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


def add_background_color(img, color):
    img = np.array(img)

    # simple background mask: dark pixels
    mask = img.mean(axis=2) < 50  

    img[mask] = color
    return Image.fromarray(img)

class CIFAR10BackgroundDataset(Dataset):
    def __init__(self, cifar_dataset, p_env):
        self.data = cifar_dataset
        self.p_env = p_env
        self.transform = T.ToTensor()
        self.vehicles = {0, 1, 8, 9}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # semantic label
        y = 0 if label in self.vehicles else 1

        # shortcut correlation
        s = y if np.random.rand() < self.p_env else 1 - y

        # background color encoding
        color = (0, 0, 255) if s == 0 else (0, 255, 0)

        img = add_background_color(img, color)
        img = self.transform(img)

        return img, y

base_train = CIFAR10(root="data", train=True, download=False)
base_test  = CIFAR10(root="data", train=False, download=False)

train_env = CIFAR10BackgroundDataset(base_train, p_env=0.9)
test_env  = CIFAR10BackgroundDataset(base_test,  p_env=0.1)

img, label = train_env[0]   # train_env = CIFAR10PatchDataset
img = img.permute(1, 2, 0)  # CHW → HWC for matplotlib

plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()