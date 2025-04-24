[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4m0h0DQR)

# 🔍 Semantic Segmentation with FCN and U-Net Variants

## 📋 Overview
This repository contains the implementation of various semantic segmentation models for deep learning-based image segmentation. The project explores different architectures including Fully Convolutional Networks (FCN) and U-Net variants. The dataset contains images and corresponding segmentation masks of size 224x224 or 256x256, depending on the task, with 13 different classes.

## 🎯 Tasks

### 1️⃣ Fully Convolutional Networks (FCN) for Semantic Segmentation

#### 1.1 Dataset Visualization 🖼️

- Visualization of the dataset by creating binary masks for each of the 13 classes:
  1. Unlabeled
  2. Building
  3. Fence
  4. Other
  5. Pedestrian
  6. Pole
  7. Roadline
  8. Road
  9. Sidewalk
  10. Vegetation
  11. Car
  12. Wall
  13. Traffic sign
- Each binary mask is displayed with appropriate titles indicating the respective class names

#### 1.2 FCN Variants 🧠

- Implementation and training of three FCN variants using a VGG16/VGG19 backbone pretrained on ImageNet:
  - **FCN-32s**: Uses only the final layer output with 32x upsampling
  - **FCN-16s**: Combines predictions from pool4 and final layer with 16x upsampling
  - **FCN-8s**: Combines predictions from pool3, pool4, and final layer with 8x upsampling
- Training scenarios:
  1. 🧊 With frozen backbone weights
  2. 🔥 With fine-tuned backbone weights
- Evaluation using Mean Intersection over Union (mIoU) and visualization of predictions

### 2️⃣ Semantic Segmentation using U-Net

#### 2.1 Vanilla U-Net 🏗️

- Implementation of the classic U-Net architecture with encoder, bottleneck, decoder, and skip connections
- Four resolution levels with progressive feature map reduction
- Training for at least 50 epochs or until convergence

#### 2.2 U-Net without Skip Connections 🔗❌

- Modified U-Net implementation without the encoder-decoder skip connections
- Comparative analysis against the standard U-Net to demonstrate the importance of skip connections

#### 2.3 Residual U-Net ➕

- Enhanced U-Net with residual convolutional blocks
- Residual blocks include two convolutional layers with a skip connection
- 1×1 convolutions in the skip path when input/output channels differ

#### 2.4 Gated Attention U-Net 👁️

- Integration of additive attention gates into the skip connections
- Analysis of how attention gates highlight salient features and suppress irrelevant regions
- Demonstration of improved segmentation precision, especially for small or detailed structures

## 📊 Evaluation Metrics

- **Mean Intersection over Union (mIoU)**: Primary metric measuring overlap between predictions and ground truth
- Loss and mIoU curves plotted for all models during training to track convergence

## 📈 Visualizations

- **Dataset Visualization**: Binary masks for each of the 13 classes
- **Training Progress**: Loss and mIoU curves throughout the training process
- **Prediction Comparison**: Predicted segmentation masks alongside ground truth images and masks

## 📚 References

- Fully Convolutional Networks for Semantic Segmentation: [arXiv:1411.4038](https://arxiv.org/abs/1411.4038) 📄
- U-Net: Convolutional Networks for Biomedical Image Segmentation: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) 📄
- Attention U-Net: Learning Where to Look for the Pancreas: [arXiv:1804.03999](https://arxiv.org/abs/1804.03999) 📄
