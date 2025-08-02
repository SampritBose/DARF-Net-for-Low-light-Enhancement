# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.slices = nn.ModuleList()

        prev_layer = 0
        for l in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_layer:l]))
            prev_layer = l

        for param in self.parameters():
            param.requires_grad = False

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, y):
        loss = 0.0
        for block in self.slices:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        return loss


class DARFLoss(nn.Module):
    def __init__(self, lambda_vae=10, lambda_gan=1, lambda_perc=0.1):
        super(DARFLoss, self).__init__()
        self.lambda_vae = lambda_vae
        self.lambda_gan = lambda_gan
        self.lambda_perc = lambda_perc

        self.perceptual_loss = PerceptualLoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def kl_divergence(self, mu, logvar):
        # logvar = log(σ²)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    def vae_loss(self, refl_pred, refl_gt, mu, logvar):
        recon_loss = self.l1_loss(refl_pred, refl_gt)
        kl_loss = self.kl_divergence(mu, logvar)
        return kl_loss + recon_loss

    def gan_discriminator_loss(self, discriminator, fake, real):
        pred_real = discriminator(real)
        pred_fake = discriminator(fake.detach())
        real_loss = self.bce_loss(pred_real, torch.ones_like(pred_real))
        fake_loss = self.bce_loss(pred_fake, torch.zeros_like(pred_fake))
        return (real_loss + fake_loss) * 0.5

    def gan_generator_loss(self, discriminator, fake):
        pred = discriminator(fake)
        return self.bce_loss(pred, torch.ones_like(pred))

    def total_loss(self, enhanced_img, target_img, refl_pred, mu, logvar, illum_pred, discriminator):
        # VAE Loss
        vae = self.vae_loss(refl_pred, target_img, mu, logvar)

        # GAN Loss
        gan = self.gan_generator_loss(discriminator, illum_pred)

        # Perceptual Loss
        perc = self.perceptual_loss(enhanced_img, target_img)

        # Combined
        total = self.lambda_vae * vae + self.lambda_gan * gan + self.lambda_perc * perc
        return total
