import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


class CustomerSegmentationNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=4, dropout_rate=0.3):
        super(CustomerSegmentationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
