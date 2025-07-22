import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dlcopilot import DLCopilot
from dlcopilot.utils import plot_learning_curves
from dlcopilot.callbacks import EarlyStopping

# Dummy CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.fc = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Data preparation
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# DLCopilot usage
copilot = DLCopilot(model, optimizer, criterion, train_loader, val_loader, device=device)

# Run basic diagnostics
copilot.inspect_input_output_shapes()
copilot.inspect_layer_shapes()
copilot.class_distribution()

# Train and visualize
copilot.train_and_analyze(epochs=2)
