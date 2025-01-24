import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np


# Define the Masked Autoencoder with Classifier
class MaskedAutoencoderWithClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MaskedAutoencoderWithClassifier, self).__init__()

        # Encoder (Masked Autoencoder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Learnable masks for MADE
        self.mask1 = nn.Parameter(torch.ones(input_dim))
        self.mask2 = nn.Parameter(torch.ones(hidden_dim))
        self.mask3 = nn.Parameter(torch.ones(latent_dim))
        self.mask4 = nn.Parameter(torch.ones(hidden_dim))

        # Classifier integrated into MADE
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x):
        mask1 = torch.sigmoid(self.mask1)
        mask2 = torch.sigmoid(self.mask2)
        mask3 = torch.sigmoid(self.mask3)
        mask4 = torch.sigmoid(self.mask4)

        h = torch.relu(self.fc1(x * mask1))
        z = torch.relu(self.fc2(h * mask2))
        h = torch.relu(self.fc3(z * mask3))
        recon = self.fc4(h * mask4)

        classification_output = self.classifier(z)
        return recon, classification_output


# Preprocess the data
def preprocess_data(input_csv, continuous_features, categorical_features, target_col):
    df = pd.read_csv(input_csv)

    # Separate continuous and categorical features
    X_continuous = df[continuous_features].values
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True).values
    y = df[target_col].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize continuous features
    scaler = StandardScaler()
    X_continuous = scaler.fit_transform(X_continuous)

    # Combine features
    X = np.hstack([X_continuous, X_categorical])
    return X, y_encoded


# Training function
def train_model(model, X_train, y_train, optimizer, reconstruction_loss_fn, classification_loss_fn, alpha, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        recon, class_output = model(X_train)

        reconstruction_loss = reconstruction_loss_fn(recon, X_train)
        classification_loss = classification_loss_fn(class_output, y_train)
        total_loss = reconstruction_loss + alpha * classification_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4f}, Classification Loss: {classification_loss.item():.4f}')


# Evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        latent_output = model.fc2(torch.relu(model.fc1(X_test * torch.sigmoid(model.mask1))))
        class_output = model.classifier(latent_output)
        preds = torch.sigmoid(class_output).cpu().numpy()
        preds_binary = (preds > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test.cpu().numpy(), preds_binary))
    print("AUC Score:")
    print(roc_auc_score(y_test.cpu().numpy(), preds))


def main(args):
    # Preprocess data
    X, y = preprocess_data(args.input_csv, args.continuous_features, args.categorical_features, args.target_col)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy().flatten()),
                                         y=y_train.numpy().flatten())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Initialize model
    model = MaskedAutoencoderWithClassifier(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim,
                                            latent_dim=args.latent_dim)

    # Optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    reconstruction_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    # Train model
    train_model(model, X_train, y_train, optimizer, reconstruction_loss_fn, classification_loss_fn, args.alpha,
                args.num_epochs)

    # Evaluate model
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Masked Autoencoder with Classifier for Binary Classification")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--continuous_features", nargs="+", required=True,
                        help="List of continuous feature column names")
    parser.add_argument("--categorical_features", nargs="+", required=True,
                        help="List of categorical feature column names")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Dimension of hidden layers")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent space")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=3.0, help="Weight for classification loss")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
