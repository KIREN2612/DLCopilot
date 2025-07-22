
### ✅ `README.md`

# DLCopilot 🚀

**DLCopilot** is your assistant for training and analyzing deep learning models in PyTorch. It provides a simple and extensible interface for visualizing model structure, input-output dimensions, training progress, and performance metrics — making your development process faster and more intuitive.

---

## 📦 Features

- 🔍 Inspect model layer-wise shapes and input-output dimensions
- 📊 Visualize class distribution in datasets
- 📈 Track and plot training and validation loss
- 🧪 Easy integration with your PyTorch training loop
- ✅ Lightweight and modular design

---

## 🛠️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KIREN2612/dlcopilot.git
cd dlcopilot
pip install -r requirements.txt
````

You can also install as a package (for development):

```bash
pip install -e .
```

---

## 🧪 Example Usage

```python
from dlcopilot import DLCopilot
from your_model import MyModel  # Replace with your model
from torch.utils.data import DataLoader

# Initialize model, optimizer, loss, data loaders
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# Create DLCopilot instance
copilot = DLCopilot(model, optimizer, criterion, train_loader, val_loader)

# Inspect architecture
copilot.inspect_input_output_shapes()
copilot.inspect_layer_shapes()

# Analyze data
copilot.class_distribution()

# Train and visualize
copilot.train_and_analyze(epochs=5)
```

See [docs/usage.md](docs/usage.md) for full examples.

---

## 🧪 Testing

Run tests with:

```bash
pytest tests/
```

Make sure to set the `PYTHONPATH` to the root of the project if needed:

```bash
export PYTHONPATH=$(pwd)  # On Linux/macOS
$env:PYTHONPATH = (Get-Location)  # On Windows PowerShell
```

---

## 📄 Documentation

* [Usage Guide](docs/usage.md)
* [API Reference (Coming Soon)]()

---

## 🧑‍💻 Author

**Kiren**
📧 [kiren2612@gmail.com](mailto:kiren2612@gmail.com)
🔗 [github.com/KIREN2612](https://github.com/KIREN2612)

---

## 📜 License

This project is licensed under the terms of the **MIT License**. See [LICENSE](LICENSE) for more details.

