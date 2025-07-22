import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from dlcopilot import DLCopilot

# Define a simple CNN model matching the example
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

@pytest.fixture
def dummy_data_loaders():
    # Create small dummy dataset of 100 samples with 1x28x28 input
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)

    # Using same for train and val just for testing
    return loader, loader

def test_dlcopilot_basic_usage(dummy_data_loaders):
    train_loader, val_loader = dummy_data_loaders
    device = 'cpu'
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    copilot = DLCopilot(model, optimizer, criterion, train_loader, val_loader, device=device)

    # Test the inspection methods run without errors
    copilot.inspect_input_output_shapes()
    copilot.inspect_layer_shapes()
    copilot.class_distribution()

    # Run one epoch of training to check train_and_analyze method
    # Use epochs=1 for fast test
    copilot.train_and_analyze(epochs=1)

    # Check that model parameters have gradients after training
    grads = [p.grad is not None for p in model.parameters()]
    assert all(grads), "All model parameters should have gradients after training"
    
