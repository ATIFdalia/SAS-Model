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

![Total Loss Formula](path_to_image/total_loss_equation.png)


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

## Customization
Hyperparameters
You can customize the following hyperparameters in the script via argparse:

-**hidden_dim: Number of neurons in the hidden layer.
-**latent_dim: Size of the latent representation.
-**alpha: Weight for the classification loss in the total loss calculation.
-**num_epochs: Number of training epochs.
-**learning_rate: Learning rate for the optimizer.


Modify these parameters directly in the script or pass them as command-line arguments when running main.py. For example:

```bash

python main.py --hidden_dim 128 --latent_dim 10 --alpha 0.5 --num_epochs 50 --learning_rate 0.001

