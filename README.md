# SAS-Model
# Smooth Adaptive Sparsity for Imbalanced Dataset (SAS)

This repository contains a PyTorch implementation of the **Smooth Adaptive Sparsity for Imbalanced Dataset (SAS)** model. SAS integrates sparsity-based feature representation and classification to handle binary classification tasks, particularly for imbalanced datasets. By leveraging a combination of smooth adaptive sparsity and cost-sensitive weighted losses, the model enhances classification performance while avoiding biases introduced by oversampling.

## Features
- **Adaptive Sparsity:** A smooth regularization mechanism encourages sparsity in the features, improving latent representation and accuracy.
- **Integrated Classifier:** The latent space is optimized for binary imbalanced classification tasks.
- **Cost-Sensitive Weighted Loss Function:** Tackles class imbalance using a cost-sensitive approach that adjusts the classification loss based on class prevalence.
- **Flexible Input:** Supports datasets with mixed continuous and categorical features.
- **Comprehensive Evaluation:** Reports key performance metrics, including classification scores and AUC-ROC.

---

## Model Architecture

The model consists of:
- **Encoder:** Encodes input data into a latent representation using learnable masks.
- **Decoder:** Reconstructs the input data from the latent representation.
- **Classifier:** Uses the latent representation to output a binary classification.

### Loss Functions
- **Reconstruction Loss:** Measures the difference between the original input and its reconstruction (`MSELoss`).
- **Classification Loss:** Cost-sensitive weighted binary cross-entropy loss (`BCEWithLogitsLoss`).

The total loss is a weighted sum of reconstruction and classification losses:

\[
\text{Total Loss} = \text{Reconstruction Loss} + \alpha \times \text{Classification Loss}
\]

---

## Installation

To set up this project, ensure the following prerequisites are installed:
- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- pandas

Install dependencies via pip:
```bash
pip install torch numpy scikit-learn pandas

