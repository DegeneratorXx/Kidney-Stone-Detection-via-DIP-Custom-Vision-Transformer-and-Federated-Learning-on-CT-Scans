# Kidney-Stone-Detection-via-DIP-Custom-Vision-Transformer-and-Federated-Learning-on-CT-Scans

## Project Overview
This project aims to develop a **Kidney Stone Detection System** using **CT Scan images** powered by a **Custom Vision Transformer (ViT)** model integrated with **Federated Learning (FL)**. The primary objective is to create a privacy-preserving solution that leverages distributed machine learning without sharing sensitive medical data between clients.

## Motivation
Kidney stone detection through medical imaging plays a crucial role in early diagnosis and treatment. However, centralized AI models often raise concerns about **data privacy** and **security** in healthcare applications. By adopting Federated Learning, the model can learn from multiple distributed sources without transferring patient data, ensuring **data confidentiality** while improving performance.

## Key Technologies
- **Custom Vision Transformer (ViT):** Used for extracting spatial features from CT Scan images.
- **Federated Learning (FedML/PySyft):** For training the model across distributed datasets without sharing raw data.
- **TensorFlow/PyTorch:** Deep learning framework for model development.
- **OpenCV & NumPy:** Image preprocessing and data augmentation (DIP).

## Custom ViT
Custom Vision Transformer Architecture
The model consists of two primary stages:

-ConvNeXt Backbone
Pre-trained ConvNeXt-Tiny model from TensorFlow Hub.
Used as the feature extractor for CT scans.
80% of layers frozen during training to leverage transfer learning.
-Vision Transformer Encoder
Multi-Head Self Attention (MHSA) layer with:
8 heads
768 embedding dimensions
Feed Forward Network (FFN) with GELU activation.
Residual connections with Layer Normalization.
Global Average Pooling for final feature embeddings.

## Features
- Privacy-preserving training using **Federated Learning**.
- Custom Vision Transformer architecture optimized for medical images.
- Data Augmentation and Preprocessing Pipeline.
- Performance Evaluation with metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Flask API for model inference.

