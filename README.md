# CAE-XGBoost for Volcanic Earthquake Classification

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-TensorFlow%20%7C%20PyTorch-orange)]()
[![ML](https://img.shields.io/badge/model-CAE%20%2B%20XGBoost-green)]()

## 📌 Overview

This project proposes a **Convolutional Autoencoder (CAE) combined with XGBoost**
for automatic classification of **volcanic seismic events**.
The CAE acts as an unsupervised feature extractor, while XGBoost performs the final classification
using latent representations.

---

## 🧠 Methodology

### Pipeline
1. Input seismic signal (2D time–frequency representation)
2. Feature extraction using CAE
3. Latent feature representation
4. Classification using XGBoost
5. Output volcanic earthquake class

### Loss Functions
- Mean Squared Error (MSE)
- Multi-domain loss (time + frequency)

---

## 📂 Dataset

- **Domain**: Volcanic seismic signals
- **Classes**:
  - VT (Volcano-Tectonic)
  - LP (Long Period)
  - Tremor
  - Hybrid (optional)

- **Preprocessing**:
  - Trimming & padding
  - Time–frequency transformation
  - Global normalization

> Dataset is not publicly available due to institutional constraints.

---

## ⚙️ Experimental Setup

### CAE
- Optimizer: Adam
- Epochs: configurable
- Latent dimension: configurable

### XGBoost
- Booster: gbtree
- Objective: multi-class classification

### Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 📊 Results

| Model         | Feature Type        | Accuracy |
|---------------|---------------------|----------|
| XGBoost       | Manual features     | 90.04%   |
| CAE-XGBoost   | CAE (MSE)           | 93.42%   |
| CAE-XGBoost   | Multi-domain loss   | 98.04%   |

---

## 📁 Project Structure

