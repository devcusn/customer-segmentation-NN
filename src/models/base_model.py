import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .neural_net import CustomerSegmentationNet
torch.manual_seed(42)
np.random.seed(42)


class CustomerSegmentationModel:
    def __init__(self, csv_file_path=None):
        self.csv_file_path = csv_file_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_preprocess_data(self, df=None):
        """Load and preprocess the customer data"""
        df = pd.read_csv(self.csv_file_path)
        print("Dataset shape:", df.shape)
        print("\nColumn names:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())

        # Handle missing values
        # Fill numerical NaNs with median
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna('Unknown')  # Fill categorical NaNs with 'Unknown'

        # Handle categorical variables (including Spending_Score and Var_1)
        categorical_columns = ['Gender', 'Ever_Married',
                               'Graduated', 'Profession', 'Spending_Score', 'Var_1']

        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(
                    f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        X = df.drop(['Segmentation'], axis=1)
        y = df['Segmentation']
        if 'ID' in X.columns:
            X = X.drop(['ID'], axis=1)

        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
            print(
                f"Target segments: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(
                    f"Warning: Column {col} is still object type, applying label encoding")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)

        print(f"\nProcessed feature matrix shape: {X_scaled.shape}")
        print(f"Feature names: {X.columns.tolist()}")

        return X_scaled, y, X.columns.tolist()

    def train_model(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        """Train the neural network model"""
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Move to device
        X_train, X_test = X_train.to(self.device), X_test.to(self.device)
        y_train, y_test = y_train.to(self.device), y_test.to(self.device)

        # Initialize model
        input_size = X_train.shape[1]
        num_classes = len(torch.unique(y))
        self.model = CustomerSegmentationNet(
            input_size, num_classes=num_classes)
        self.model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        train_losses = []
        train_accuracies = []

        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_train).float().mean()

            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())

            if (epoch + 1) % 20 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test).float().mean()

        print(f'\nTest Accuracy: {test_accuracy.item():.4f}')

        # Plot training history
        self.plot_training_history(train_losses, train_accuracies)

        return X_test.cpu(), y_test.cpu(), test_predicted.cpu()

    def plot_training_history(self, losses, accuracies):
        """Plot training loss and accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(accuracies)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)

        return predicted.cpu().numpy(), probabilities.cpu().numpy()

    def get_feature_importance(self, feature_names):
        """Get feature importance based on model weights"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get weights from first layer
        weights = self.model.fc1.weight.data.cpu().numpy()
        feature_importance = np.abs(weights).mean(axis=0)

        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df
