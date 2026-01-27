from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


def add_patch(img, color, patch_size=6):
    img = img.copy()
    patch = Image.new("RGB", (patch_size, patch_size), color)
    img.paste(patch, (0, 0))  # top-left
    return img


class CIFAR10PatchDataset(Dataset):
    def __init__(self, cifar_dataset, p_env):
        """
        cifar_dataset : torchvision.datasets.CIFAR10
        p_env          : probability that patch matches label
        """
        self.data = cifar_dataset
        self.p_env = p_env
        self.transform = T.ToTensor()

        # vehicles vs animals
        self.vehicles = {0, 1, 8, 9}   # airplane, auto, ship, truck

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # binary semantic label
        y = 0 if label in self.vehicles else 1

        # shortcut label
        if np.random.rand() < self.p_env:
            s = y
        else:
            s = 1 - y

        # encode shortcut with color
        color = (255, 0, 0) if s == 0 else (0, 255, 0)

        img = add_patch(img, color)
        img = self.transform(img)

        return img, y
    


base_train = CIFAR10(root="data", train=True, download=False)
base_test  = CIFAR10(root="data", train=False, download=False)

train_env = CIFAR10PatchDataset(base_train, p_env=0.9)
test_env  = CIFAR10PatchDataset(base_test,  p_env=0.1)

img, label = train_env[0]   # train_env = CIFAR10PatchDataset
img = img.permute(1, 2, 0)  # CHW → HWC for matplotlib

plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()