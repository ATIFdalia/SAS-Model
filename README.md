# SAS-Model
# **Smooth Adaptive Sparsity for Imbalanced Dataset (SAS)**

This repository contains a PyTorch implementation of the **Smooth Adaptive Sparsity for Imbalanced Dataset (SAS)** model. SAS integrates sparsity-based feature representation and classification to handle binary classification tasks, particularly for imbalanced datasets. By leveraging a combination of smooth adaptive sparsity and weighted losses, the model enhances classification performance while avoiding biases introduced by oversampling.

---

## **Features**
- **Adaptive Sparsity:** A smooth regularization mechanism encourages sparsity in the features, improving latent representation and accuracy.
- **Integrated Classifier:** The latent space is optimized for binary imbalanced classification tasks.
- **Weighted Loss Function:** Tackles class imbalance by applying weighted classification loss to penalize the minority class effectively.
- **Flexible Input:** Supports datasets with mixed continuous and categorical features.
- **Comprehensive Evaluation:** Reports key performance metrics, including classification scores, AUC-ROC, and detailed classification reports.

---

## **Installation**
To set up this project, ensure the following prerequisites are installed:
- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- pandas

You can install the required dependencies using pip:
```bash
pip install torch numpy scikit-learn pandas

