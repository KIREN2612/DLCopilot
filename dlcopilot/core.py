# dlcopilot/core.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from .utils import plot_learning_curves, show_confusion_matrix

class DLCopilot:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"\n✅ DLCopilot initialized successfully on device: {self.device}")

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def inspect_input_output_shapes(self):
        x, y = next(iter(self.train_loader))
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model(x)
        print("\n🔍 Inspecting Input and Output Shapes:")
        print(f"📥 Input Shape: {x.shape}")
        print(f"📤 Output Shape: {out.shape}")

    def class_distribution(self):
        class_counts = defaultdict(int)
        for _, y in self.train_loader:
            for label in y:
                class_counts[int(label)] += 1
        print("\n📊 Checking Class Distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f" - Class {cls}: {count} samples")
        total = sum(class_counts.values())
        if max(class_counts.values()) - min(class_counts.values()) > 0.2 * total / len(class_counts):
            print("⚠️ Class distribution may be imbalanced.")
        else:
            print("✅ Class distribution looks balanced.")

    def inspect_layer_shapes(self, input_shape=(1, 1, 28, 28)):
        print("\n🧠 Inspecting Layer-wise Output Shapes:")
        x = torch.randn(input_shape).to(self.device)
        hooks = []

        def register_hook(module):
            if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != self.model:
                def hook(module, input, output):
                    class_name = module.__class__.__name__
                    output_shape = tuple(output.shape)
                    print(f"🔸 {class_name} → Output Shape: {output_shape}")
                hooks.append(module.register_forward_hook(hook))

        self.model.apply(register_hook)
        with torch.no_grad():
            self.model(x)
        for h in hooks:
            h.remove()

    def check_gradients(self):
        self.model.train()
        x, y = next(iter(self.train_loader))
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()

        total_norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                total_norms.append(p.grad.data.norm(2).item())

        if total_norms:
            print("\n📉 Checking for Vanishing/Exploding Gradients:")
            print(f"📏 Gradient Norms — Mean: {np.mean(total_norms):.4f}, Max: {np.max(total_norms):.4f}, Min: {np.min(total_norms):.4f}")
            if np.max(total_norms) > 1000:
                print("⚠️ Potential exploding gradients.")
            elif np.min(total_norms) < 1e-6:
                print("⚠️ Potential vanishing gradients.")
            else:
                print("✅ Gradient norms look healthy.")
        else:
            print("⚠️ No gradients found. Was there a forward and backward pass?")

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            _, pred = torch.max(out, 1)
        return pred.cpu().numpy()

    def train_and_analyze(self, epochs=5):
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            for x, y in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = total_loss / len(self.train_loader)
            train_acc = correct / total

            val_loss, val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"\n=== Epoch {epoch} ===")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if epoch >= 5:
                if train_loss < val_loss:
                    print("⚠️ Possible Overfitting detected: Training loss decreased but Validation loss increased.")
                else:
                    print("✅ No obvious overfitting or underfitting detected.")
            else:
                print("ℹ️ Need at least 6 epochs to detect plateau.")

            self.check_gradients()

        plot_learning_curves(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)
        show_confusion_matrix(self.model, self.val_loader, self.device)

    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                _, preds = torch.max(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / len(self.val_loader), correct / total
