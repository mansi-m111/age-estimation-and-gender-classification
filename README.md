# 👵🧑 Age Estimation and Gender Classification Using CNNs

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)

This project implements and compares two Convolutional Neural Network (CNN) models to predict **age** and **gender** from face images. Using a subset of the **UTKFace dataset** (5,000 images of individuals aged 0 to 116), the models perform simultaneous:

- Gender Classification (Binary)
- Age Estimation (Regression)

One model is built from scratch (Model A), and the other fine-tunes a pre-trained VGG16 network (Model B). The goal is to evaluate the performance of custom and transfer learning approaches for this dual-task problem.

---

## 📌 Project Overview

- **Dataset**: UTKFace Subset (5,000 images, balanced gender distribution)
- **Tasks**:
  - Gender Classification: Binary (0 = Male, 1 = Female)
  - Age Estimation: Regression (continuous value)
- **Models**:
  - **Model A**: Custom CNN with 4 convolutional blocks and dual output branches
  - **Model B**: Fine-tuned VGG16 with frozen base layers and custom dense heads
- **Evaluation Metrics**:
  - Accuracy (for gender classification)
  - Mean Absolute Error – MAE (for age estimation)

---

## 🎯 Objectives

- Build a CNN from scratch for simultaneous gender and age prediction.
- Fine-tune a pre-trained VGG16 model for the same dual task.
- Compare model performance on training/validation sets.
- Analyze learning curves for overfitting and convergence insights.

---

## 🧰 Tools & Technologies

- **Languages**: Python  
- **Libraries**:
  - `TensorFlow` / `Keras`
  - `Pandas`, `NumPy`
  - `Matplotlib`
- **Environment**: Jupyter Notebook

---

## 🧪 Methodology

### Data Preprocessing

- Extracted age and gender labels from image filenames.
- Split data into 80% training and 20% validation.
- Applied data augmentation:
  - Rotation, shifts, shear, zoom, horizontal flip
- Resized all images to **128x128 pixels**.
  
---

### Model A: Custom CNN

- **Architecture**:
  - 4 convolutional blocks with Batch Normalization + MaxPooling
  - Two output branches:
    - Sigmoid for gender classification
    - Linear for age regression
- **Training**:
  - Optimizer: Adam
  - Loss:
    - Binary Cross-Entropy (gender)
    - Mean Squared Error (age)
  - Regularization:
    - Dropout (30%)
    - Early stopping to prevent overfitting

---

### Model B: Fine-tuned VGG16

- Used VGG16 pre-trained on ImageNet.
- **Frozen** first 14 layers.
- Added custom dense heads for age and gender.
- Similar training setup as Model A.

---

## 📊 Key Results

| Metric                          | Model A (Custom CNN) | Model B (VGG16 Fine-tuned) |
|---------------------------------|----------------------|-----------------------------|
| **Gender Classification Accuracy (Val)** | 85.8%               | 82.6%                      |
| **Age Estimation MAE (Val)**           | 7.3 years           | 6.9 years
| **Training Epochs**                    | 40                  | 30                         |

- **Model A** achieved **higher gender classification accuracy**, but with slightly **higher age estimation error**.
- **Model B** demonstrated **more stable training**, converging faster and producing better age predictions.
- Both models confirm that **age estimation is a harder problem** than gender classification.
- Regularization techniques like **early stopping and dropout** effectively prevented overfitting.
