# 🧐 Customer Segmentation using Neural Networks

This repository contains a simple yet effective neural network built with **PyTorch** for customer segmentation based on demographic and behavioral features.

## 📦 Project Structure

```
customer-segmentation-NN/
│
├── data/                        # Dataset location (after download)
│   └── external/
│       └── customer.zip
│
├── models/
│   ├── base_model.py           # CustomerSegmentationModel class
│   ├── neural_net.py           # PyTorch neural network architecture
│   └── init.py                 # Main script for training and evaluation
│
├── utils/
│   ├── get_data.py             # Download utility
│   └── zip.py                  # ZIP extraction utility
│
├── run.py                      # Entry point to automate the full pipeline
└── README.md                   # Project documentation
```

## 📊 Features

- Downloads customer segmentation dataset from Kaggle
- Preprocesses demographic and behavioral features
- Encodes categorical variables
- Trains a feedforward neural network with dropout and ReLU
- Evaluates performance (accuracy, classification report)
- Computes basic feature importance based on model weights

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-segmentation-NN.git
cd customer-segmentation-NN
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python src/init.py
```

This will:

- Download the dataset
- Extract it
- Train the model
- Display evaluation results and feature importance

## 🧠 Neural Network Architecture

- Input → Linear(128) → ReLU → Dropout
- → Linear(64) → ReLU → Dropout
- → Linear(output classes)

## 📈 Sample Output

- Training Loss and Accuracy plots
- Test Accuracy
- Classification Report
- Feature Importance Ranking
- Sample Predictions with Probabilities

## 📚 Dataset

Dataset source: [Kaggle - Customer Segmentation Dataset](https://www.kaggle.com/datasets/vetrirah/customer)

**Features include:**

- Demographic: Gender, Age, Married status, Education, Profession, etc.
- Behavioral: Spending Score, Work Experience

## 📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and contribute!

---
