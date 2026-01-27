from torchvision.datasets import CIFAR10
from texture_shortcut import CIFAR10TextureDataset
from bg_shortcut import CIFAR10BackgroundDataset
from watermark import CIFAR10PatchDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


p_train = 0.9   # shortcut mostly correct
p_test  = 0.1   # shortcut mostly wrong (OOD)

class CIFAR10NormalDataset(Dataset):
    """
    Clean CIFAR-10 binary classification dataset (vehicles vs animals).
    No shortcuts applied.
    """

    def __init__(self, cifar_dataset):
        """
        cifar_dataset: torchvision.datasets.CIFAR10
        """
        self.data = cifar_dataset
        self.transform = T.ToTensor()

        # vehicle classes
        self.vehicles = {0, 1, 8, 9}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # binary label
        y = 0 if label in self.vehicles else 1
        img = img.copy()
        img = self.transform(img)
        return img, y

base_train = CIFAR10(root="./data", train=True, download=False)
base_test  = CIFAR10(root="./data", train=False, download=False)

train_dataset_normal = CIFAR10NormalDataset(base_train)
test_dataset_normal  = CIFAR10NormalDataset(base_test)

train_dataset_patch = CIFAR10PatchDataset(base_train, p_env=p_train)
test_dataset_patch  = CIFAR10PatchDataset(base_test,  p_env=p_test)

train_dataset_bg = CIFAR10BackgroundDataset(base_train, p_env=p_train)
test_dataset_bg  = CIFAR10BackgroundDataset(base_test,  p_env=p_test)

train_dataset_texture = CIFAR10TextureDataset(base_train, p_env=p_train)
test_dataset_texture  = CIFAR10TextureDataset(base_test,  p_env=p_test)



train_loader_patch = DataLoader(
    train_dataset_patch,
    batch_size=128,
    shuffle=True
)

test_loader_patch = DataLoader(
    test_dataset_patch,
    batch_size=128,
    shuffle=False
)

train_loader_normal = DataLoader(
    train_dataset_normal,
    batch_size=128,
    shuffle=True
)

test_loader_normal = DataLoader(
    test_dataset_normal,
    batch_size=128,
    shuffle=False
)

train_loader_bg = DataLoader(
    train_dataset_bg,
    batch_size=128,
    shuffle=True
)

test_loader_bg = DataLoader(
    test_dataset_bg,
    batch_size=128,
    shuffle=False
)

train_loader_texture = DataLoader(
    train_dataset_texture,
    batch_size=128,
    shuffle=True
)

test_loader_texture = DataLoader(
    test_dataset_texture,
    batch_size=128,
    shuffle=False
)


def get_resnet18():
    model = models.resnet18(weights=None)

    # Modify for CIFAR-10
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    # Binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_resnet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_accs_normal= []
test_accs_normal = []
epochs = 10

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader_normal)
    test_acc = evaluate(model, test_loader_normal)

    print(
        f"Epoch {epoch+1}: "
        f"Train Acc = {train_acc:.3f}, "
        f"Test Acc = {test_acc:.3f}"
    )
    train_accs_normal.append(train_acc)
    test_accs_normal.append(test_acc)
torch.save(model, "checkpoints/resnet18_normal.pth")

epochs_range = range(1, len(train_accs_normal) + 1)

plt.figure()
plt.plot(epochs_range, train_accs_normal, label="Train Accuracy")
plt.plot(epochs_range, test_accs_normal, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy(normal)")
plt.legend()
plt.grid(True)
plt.show()

train_accs_patch= []
test_accs_patch = []
epochs = 10

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader_patch)
    test_acc = evaluate(model, test_loader_patch)

    print(
        f"Epoch {epoch+1}: "
        f"Train Acc = {train_acc:.3f}, "
        f"Test Acc = {test_acc:.3f}"
    )
    train_accs_patch.append(train_acc)
    test_accs_patch.append(test_acc)

torch.save(model, "checkpoints/resnet18_patch.pth")

epochs_range = range(1, len(train_accs_patch) + 1)

plt.figure()
plt.plot(epochs_range, train_accs_patch, label="Train Accuracy")
plt.plot(epochs_range, test_accs_patch, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy(patch)")
plt.legend()
plt.grid(True)
plt.show()

train_accs_texture= []
test_accs_texture = []
epochs = 10

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader_texture)
    test_acc = evaluate(model, test_loader_texture)

    print(
        f"Epoch {epoch+1}: "
        f"Train Acc = {train_acc:.3f}, "
        f"Test Acc = {test_acc:.3f}"
    )
    train_accs_texture.append(train_acc)
    test_accs_texture.append(test_acc)

torch.save(model, "checkpoints/resnet18_texture.pth")


epochs_range = range(1, len(train_accs_texture) + 1)

plt.figure()
plt.plot(epochs_range, train_accs_texture, label="Train Accuracy")
plt.plot(epochs_range, test_accs_texture, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy(texture)")
plt.legend()
plt.grid(True)
plt.show()

train_accs_bg= []
test_accs_bg = []
epochs = 10

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader_bg)
    test_acc = evaluate(model, test_loader_bg)

    print(
        f"Epoch {epoch+1}: "
        f"Train Acc = {train_acc:.3f}, "
        f"Test Acc = {test_acc:.3f}"
    )
    train_accs_bg.append(train_acc)
    test_accs_bg.append(test_acc)
torch.save(model, "resnet18_bg.pth")


epochs_range = range(1, len(train_accs_bg) + 1)

plt.figure()
plt.plot(epochs_range, train_accs_bg, label="Train Accuracy")
plt.plot(epochs_range, test_accs_bg, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy(bg)")
plt.legend()
plt.grid(True)
plt.show()




