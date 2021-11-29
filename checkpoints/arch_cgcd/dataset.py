import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.resize(self.images[idx], (32, 32))
        return self.transforms(image), self.labels[idx]
