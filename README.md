# DARF-Net-for-Low-light-Enhancement
**DARF-Net: A Dual Attention Retinex-based Fusion Network for Low Light Image Enhancement**

**Authors: Samprit Bose, Agnesh Chandra Yadav, and Maheshkumar H. Kolekar**

# Abstract

Low-light images, which are taken under inadequate lighting conditions utilizing typical imaging sensors, frequently have low contrast, poor visibility, and features lost in areas that are not well lit. These inherent problems, along with the possibility of artifacts, overexposure, or insufficient detail preservation, especially in complicated lighting environments, make it difficult to improve such images. To overcome these difficulties, we proposed DARF-Net, a novel framework based on Retinex theory, which decomposes input images into illumination and reflectance components.Our method, which uses an Illumination-Guided Multi-Head Attention module as the generator of a generative adversarial network, improves the illumination map, while a Variational Autoencoder supplemented with a Spatial Attention Module improves the reflectance map. The combined effect of these two attention mechanisms enhances perceived quality, brightness, and structural faithfulness. Comprehensive tests on the LOL and SICE datasets show that DARF-Net outperforms the state-of-the-art techniques in terms of PSNR, SSIM, and LPIPS metrics while retaining computational efficiency, which makes it a good fit for real-world deployment.

# Methodology

![Block_Final](https://github.com/user-attachments/assets/18c19af7-9336-4dd9-b003-08c82e458c57)


