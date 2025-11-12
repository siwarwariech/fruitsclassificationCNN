# Fruit Classification using Image Analysis
# 🍎 Fruit Quality Classification Dashboard

A comprehensive deep learning dashboard for classifying fruit quality (Fresh vs Rotten) using multiple CNN architectures with interactive visualization.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Interactive-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%2FResNet-orange)

## 🌟 Features

- **Multi-Model Comparison**: CNN, ResNet, and CNN with Attention (CBAM)
- **Interactive Dashboard**: Built with Plotly Dash and Bootstrap
- **Real-time Prediction**: Upload images for instant classification
- **Comprehensive Analytics**: Confusion matrices, ROC curves, F1 scores
- **Visual Interpretations**: Model performance comparisons and error analysis
- **Responsive Design**: Material Design inspired interface

## 📊 Project Overview

Fruit Quality Classifier is a comprehensive deep learning application that provides:

- [x] **Multi-Model Comparison** - CNN, ResNet, and CNN+Attention
- [x] **Real-time Prediction** - Upload images for instant classification
- [x] **Interactive Dashboard** - Built with Plotly Dash and Bootstrap
- [x] **Comprehensive Analytics** - Confusion matrices, ROC curves, F1 scores
- [x] **Model Interpretations** - Performance comparisons and recommendations
- [ ] **Mobile App Integration** - Future development
## 🏗️ Project Architecture

```
fruit-classification/
├── Dashboard Application
│   ├── app.py                 # Main Dash application
│   ├── requirements.txt       # Python dependencies
│   └── static/               # CSS and assets
├── Model Analysis
│   ├── notebooks/            # Jupyter notebooks
│   │   └── fruit_classification.ipynb
│   └── models/              # Trained model weights
├── Data & Assets
│   ├── data/                # Dataset and metadata
│   └── assets/              # Images and static files
└── Documentation
    ├── README.md            # Project documentation
    ├── LICENSE              # MIT License
    └── .gitignore           # Git exclusion rules
```


## 🧠 Technical Details
Deep Learning Architectures
1. CNN (Convolutional Neural Network)
Standard convolutional architecture with multiple layers

Fast inference suitable for real-time applications

Excellent baseline for performance comparison

2. ResNet (Residual Network)
Pre-trained ResNet50 with transfer learning

Deep architecture with skip connections for gradient flow

Higher accuracy but requires more computational resources

3. CNN + CBAM (Convolutional Block Attention Module)
Custom CNN integrated with attention mechanism

Channel and spatial attention for feature refinement


Optimal balance between accuracy and inference speed

## 🔄 Application Workflow

```mermaid
graph TD
    A[User Uploads Fruit Image] --> B[Image Preprocessing]
    B --> C[Multi-Model Prediction]
    C --> D[CNN Model Analysis]
    C --> E[ResNet Model Analysis]
    C --> F[CNN+Attention Analysis]
    D --> G[Results Comparison]
    E --> G
    F --> G
    G --> H[Interactive Visualization]
    G --> I[Performance Metrics]
    G --> J[Download Results]
