import kagglehub
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os

def data_ready():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    train_dir = os.path.join(path, "chest_xray", "train")
    val_dir = os.path.join(path, "chest_xray", "val")
    test_dir = os.path.join(path, "chest_xray", "test")
    return train_dir, val_dir, test_dir

def data_loader_set(train_dir, val_dir, test_dir):
    transforms = {
        'train': v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.RandomVerticalFlip(p=0.3),
            v2.ColorJitter(brightness=0.3, saturation=0.3),
            v2.RandomPerspective(distortion_scale=0.3, p=0.3),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            v2.ToDtype(dtype=torch.float32, scale=True)
            ]),
        'val': v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
            ]),
        'test': v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
            ])}

    train_dataset = datasets.ImageFolder(train_dir, transform=transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms['test'])

    extra_val_size = len(test_dataset) - len(val_dataset)
    train_size = len(train_dataset) - extra_val_size

    train_dataset_modify, extra_val_dataset = torch.utils.data.random_split(train_dataset, [train_size, extra_val_size])

    val_dataset = torch.utils.data.ConcatDataset([val_dataset, extra_val_dataset])

    train_loader = DataLoader(train_dataset_modify, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names