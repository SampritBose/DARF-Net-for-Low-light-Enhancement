# test.py
import torch
from torch.utils.data import DataLoader
from dataset import PairedImageDataset
from model import DARFNet
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset
test_dir = r"........"
test_dataset = PairedImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = DARFNet().to(device)
model.load_state_dict(torch.load("checkpoints/darf_net.pth"))
model.eval()

# Output directory
os.makedirs("results", exist_ok=True)

# Inference
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        input_img = data["low"].to(device)
        enhanced, _, _, _, _ = model(input_img)

        save_path = f"results/output_{idx}.png"
        vutils.save_image(enhanced, save_path)

        print(f"Saved: {save_path}")
