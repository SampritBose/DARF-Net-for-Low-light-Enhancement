# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.low_dir = os.path.join(root_dir, "low")  # or 'input'
        self.high_dir = os.path.join(root_dir, "high")  # or 'gt'
        self.transform = transform

        self.low_images = sorted(os.listdir(self.low_dir))
        self.high_images = sorted(os.listdir(self.high_dir))

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_dir, self.low_images[idx])
        high_img_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img = Image.open(low_img_path).convert("RGB")
        high_img = Image.open(high_img_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return {"low": low_img, "high": high_img}
