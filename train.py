# train.py
import torch
from torch.utils.data import DataLoader
from dataset import PairedImageDataset
from model import DARFNet  # From previous completion
from loss import DARFLoss  # Also includes GAN, VAE, Perceptual
import torchvision.transforms as transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset
train_dir = r"......."
train_dataset = PairedImageDataset(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model
model = DARFNet().to(device)
discriminator = Discriminator().to(device)  # As defined before

# Optimizers
optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.05)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

# Loss
criterion = DARFLoss()

# Training loop
for epoch in range(150):
    model.train()
    for i, data in enumerate(train_loader):
        input_img = data["low"].to(device)
        target_img = data["high"].to(device)

        # Forward
        enhanced, illum_pred, refl_pred, mu, logvar = model(input_img)

        # === Discriminator ===
        optimizer_D.zero_grad()
        d_loss = criterion.gan_discriminator_loss(discriminator, illum_pred.detach(), target_img)
        d_loss.backward()
        optimizer_D.step()

        # === Generator ===
        optimizer_G.zero_grad()
        g_loss = criterion.total_loss(enhanced, target_img, refl_pred, mu, logvar, illum_pred, discriminator)
        g_loss.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/150] Batch [{i}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

# Save
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/darf_net.pth")
