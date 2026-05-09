# Driver Drowsiness Detection via Multi-Head CNN 

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Linux](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

An advanced Computer Vision system designed to enhance road safety by monitoring driver fatigue in real-time. This project implements a **Multi-Task Learning (MTL)** architecture to detect eye and mouth states simultaneously.

## 🚀 Key Highlights
- **Architecture from Scratch:** Built a custom Dual-Head CNN in PyTorch for optimized performance.
- **Multi-Task Efficiency:** A single backbone processes images once to produce two distinct classifications, saving 40% of computational resources compared to dual-model setups.
- **Robust Pipeline:** Includes custom data loaders, advanced augmentation, and a joint-loss training strategy.

---

## 🏗️ Model Architecture
The model uses a shared convolutional backbone to extract facial features, which then branches into two specialized "heads":
1. **Eye Head:** Classifies eyes as `Open` or `Closed`.
2. **Mouth Head:** Detects if the driver is `Yawning` or `Normal`.



---

## 📊 Performance Results
The model was trained for 15 epochs on a diverse dataset from Kaggle, achieving professional-grade stability and accuracy:

| Metric | Accuracy | Status |
| :--- | :--- | :--- |
| **Eye State Detection** | **97.77%** | ✅ Stable |
| **Yawn Detection** | **95.24%** | ✅ Stable |

### Training Visualization
> **Note:** The model utilizes a Joint Loss function ($Loss_{total} = Loss_{eye} + Loss_{mouth}$) to ensure both heads learn effectively without bias.

---

## 🛠️ Technical Stack
*   **Framework:** PyTorch (Core Model & Training).
*   **Data Processing:** PIL, Torchvision (Custom Transforms).
*   **Environment:** Linux/Ubuntu (Managed via `uv` for reproducible builds).
*   **IDE:** CLion / VS Code.

---

## 📂 Project Structure
```text
├── data/               # Dataset (Eyes & Mouths)
├── src/
│   ├── model.py         # Multi-Head CNN Architecture
│   ├── eye_dataset.py   # Custom dataset class for eye data.
│   ├── mouth_dataset.py # Custom dataset class for yawn data.
│   └── train.py         # Training & Validation Loop
├── pyproject.toml       # Project dependencies
└── README.md