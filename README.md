# DARF-Net-for-Low-light-Enhancement
**DARF-Net: A Dual Attention Retinex-based Fusion Network for Low Light Image Enhancement**

**Authors: Samprit Bose, Agnesh Chandra Yadav, and Maheshkumar H. Kolekar**

# Abstract

<p>
Low-light images, which are taken under inadequate lighting conditions utilizing typical imaging sensors, frequently have low contrast, poor visibility, and features lost in areas that are not well lit. These inherent problems, along with the possibility of artifacts, overexposure, or insufficient detail preservation, especially in complicated lighting environments, make it difficult to improve such images. To overcome these difficulties, we proposed DARF-Net, a novel framework based on Retinex theory, which decomposes input images into illumination and reflectance components.Our method, which uses an Illumination-Guided Multi-Head Attention module as the generator of a generative adversarial network, improves the illumination map, while a Variational Autoencoder supplemented with a Spatial Attention Module improves the reflectance map. The combined effect of these two attention mechanisms enhances perceived quality, brightness, and structural faithfulness. Comprehensive tests on the LOL and SICE datasets show that DARF-Net outperforms the state-of-the-art techniques in terms of PSNR, SSIM, and LPIPS metrics while retaining computational efficiency, which makes it a good fit for real-world deployment.
</p>

# Methodology

![Block_Final](https://github.com/user-attachments/assets/18c19af7-9336-4dd9-b003-08c82e458c57)

### Main Components:
- **Decomposition Module**: Splits input into Illumination (L) and Reflectance (R).
- **Illumination Enhancement**: IGMHA + GAN improves L under poor lighting.
- **Reflectance Enhancement**: VAE + Spatial Attention refines R for structure and texture.
- **Reconstruction**: Combines I' = L' × R'

### Total Loss Function:
DARF-Net optimizes a combined loss function:
L_total = λ1 * L_VAE + λ2 * L_GAN + λ3 * L_Perc

- λ1 = 10, 
- λ2 = 1, 
- λ3 = 0.1
  
These weights ensure optimal balance between realism, structure preservation, and low-level detail.

# Results

![SOTA](https://github.com/user-attachments/assets/3ec3dc8b-282a-44d3-99c5-664a3c5857b0)


# Performance

![Table](https://github.com/user-attachments/assets/94474779-8434-4d40-bf21-0fc7449d31be)

# Cite as



