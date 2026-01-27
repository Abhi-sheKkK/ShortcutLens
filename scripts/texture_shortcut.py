from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

def add_texture(img, texture_type="vertical"):
    img = np.array(img).astype(np.float32)

    h, w, _ = img.shape

    if texture_type == "vertical":
        for x in range(0, w, 4):
            img[:, x:x+2] += 30
    else: 
        for y in range(0, h, 4):
            img[y:y+2, :] += 30

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

class CIFAR10TextureDataset(Dataset):
    def __init__(self, cifar_dataset, p_env):
        self.data = cifar_dataset
        self.p_env = p_env
        self.transform = T.ToTensor()
        self.vehicles = {0, 1, 8, 9}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        y = 0 if label in self.vehicles else 1
        s = y if np.random.rand() < self.p_env else 1 - y

        texture = "vertical" if s == 0 else "horizontal"
        img = add_texture(img, texture)

        img = self.transform(img)
        return img, y
    
base_train = CIFAR10(root="data", train=True, download=False)
base_test  = CIFAR10(root="data", train=False, download=False)

train_env = CIFAR10TextureDataset(base_train, p_env=0.9)
test_env  = CIFAR10TextureDataset(base_test,  p_env=0.1)

img, label = train_env[10]   # train_env = CIFAR10PatchDataset
img = img.permute(1, 2, 0)  # CHW → HWC for matplotlib

plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

