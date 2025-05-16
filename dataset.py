import torch
from torch.utils.data import Dataset
from model import CNNClassifier
from train import train_image_model
from torch import nn


images = torch.randint(0, 256, (100, 3, 50, 50), dtype=torch.uint8)
labels = torch.randint(0, 2, (100,), dtype=torch.long)


class ImageDataset(Dataset):
    def __init__(self, data, labels, n_channels=3):

        self.data= data
        self.labels=labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

img_dataset = ImageDataset(images,labels)

model = CNNClassifier(n_channels=3, n_classes=2)

train_image_model(model=model,dataset=img_dataset)
