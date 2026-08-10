#!/usr/bin/env python3
"""
Train a simple 2-layer fully connected network on MNIST and export weights
as raw binary files for InferX C++ inference.

Model architecture:
    Input (784) → Linear(784, 128) → ReLU → Linear(128, 10) → Softmax

Expected accuracy: ~97% on MNIST test set.

Weight export format (.bin):
    - Header: 4 bytes magic "INFX"
    - 4 bytes uint32: number of layers
    - For each layer:
        - 4 bytes uint32: rows (output_dim)
        - 4 bytes uint32: cols (input_dim)
        - rows * cols * 4 bytes: weight matrix (float32, row-major)
        - rows * 4 bytes: bias vector (float32)

Usage:
    pip install torch torchvision
    python scripts/export_mnist_weights.py

Output:
    models/mnist_fc_weights.bin
    models/mnist_test_images.bin   (first 100 test images for C++ validation)
    models/mnist_test_labels.bin   (corresponding labels)
"""

import struct
import os
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
except ImportError:
    print("ERROR: PyTorch and torchvision required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)


class MNISTNet(nn.Module):
    """Simple 2-layer FC network for MNIST."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(epochs=5, lr=0.001, batch_size=64):
    """Train the MNIST model."""
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    model = MNISTNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

    print(f"\nFinal test accuracy: {accuracy:.2f}%")
    return model, test_dataset


def export_weights(model, output_path):
    """Export model weights in InferX binary format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    layers = [
        (model.fc1.weight.data, model.fc1.bias.data),
        (model.fc2.weight.data, model.fc2.bias.data),
    ]

    with open(output_path, 'wb') as f:
        # Magic header
        f.write(b'INFX')
        # Number of layers
        f.write(struct.pack('<I', len(layers)))

        for weight, bias in layers:
            rows, cols = weight.shape
            f.write(struct.pack('<I', rows))
            f.write(struct.pack('<I', cols))
            # Weight matrix (row-major float32)
            f.write(weight.numpy().tobytes())
            # Bias vector
            f.write(bias.numpy().tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Exported weights to {output_path} ({file_size / 1024:.1f} KB)")


def export_test_data(test_dataset, output_dir, num_samples=100):
    """Export test images and labels for C++ validation."""
    os.makedirs(output_dir, exist_ok=True)

    images_path = os.path.join(output_dir, 'mnist_test_images.bin')
    labels_path = os.path.join(output_dir, 'mnist_test_labels.bin')

    with open(images_path, 'wb') as f_img, open(labels_path, 'wb') as f_lbl:
        # Header: number of samples, image size
        f_img.write(struct.pack('<II', num_samples, 784))
        f_lbl.write(struct.pack('<I', num_samples))

        for i in range(num_samples):
            img, label = test_dataset[i]
            # Normalize same as training: (x - 0.1307) / 0.3081
            # torchvision already did this in the transform
            flat = img.view(784).numpy()
            f_img.write(flat.tobytes())
            f_lbl.write(struct.pack('<B', label))

    print(f"Exported {num_samples} test samples to {output_dir}/")


if __name__ == '__main__':
    # Train
    model, test_dataset = train_model(epochs=5)

    # Export
    export_weights(model, 'models/mnist_fc_weights.bin')
    export_test_data(test_dataset, 'models/', num_samples=100)

    print("\nDone! Run the C++ inference with:")
    print("  ./build-release/examples/mnist_inference models/")
