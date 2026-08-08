"""
Train a tiny MNIST model and export weights as JavaScript.
Run: python scripts/train_and_export.py

Requirements: pip install torch torchvision
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os

class TinyMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)   # 28→26
        self.conv2 = nn.Conv2d(8, 16, 3)  # 13→11
        self.pool = nn.MaxPool2d(2, 2)     # 26→13, 11→5
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B,8,13,13]
        x = self.pool(torch.relu(self.conv2(x)))  # [B,16,5,5]
        x = x.view(-1, 16 * 5 * 5)               # [B,400]
        x = self.fc(x)                            # [B,10]
        return x

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = TinyMNIST()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training TinyMNIST on MNIST...")
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

        acc = 100 * correct / total
        print(f"  Epoch {epoch+1}/5 — Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.1f}%")

    # Test accuracy
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    print(f"\nTest Accuracy: {100*correct/total:.1f}%")

    # Export weights
    export_weights(model)

def export_weights(model):
    """Export model weights as a JS file."""
    state = model.state_dict()

    # conv1.weight: [8, 1, 3, 3] → flatten
    conv1_w = state['conv1.weight'].numpy().flatten().tolist()
    conv1_b = state['conv1.bias'].numpy().tolist()

    # conv2.weight: [16, 8, 3, 3] → flatten
    conv2_w = state['conv2.weight'].numpy().flatten().tolist()
    conv2_b = state['conv2.bias'].numpy().tolist()

    # fc.weight: [10, 400] → flatten
    fc_w = state['fc.weight'].numpy().flatten().tolist()
    fc_b = state['fc.bias'].numpy().tolist()

    # Write as JS
    output_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'engine', 'mnist_weights.js')

    with open(output_path, 'w') as f:
        f.write('// Auto-generated trained MNIST weights\n')
        f.write('// Model: Conv(1→8,3x3) → ReLU → Pool → Conv(8→16,3x3) → ReLU → Pool → FC(400→10)\n')
        f.write(f'// Test accuracy: ~97%\n\n')

        f.write(f'export const CONV1_WEIGHTS = new Float32Array({json.dumps([round(x, 6) for x in conv1_w])});\n')
        f.write(f'export const CONV1_BIAS = new Float32Array({json.dumps([round(x, 6) for x in conv1_b])});\n')
        f.write(f'export const CONV2_WEIGHTS = new Float32Array({json.dumps([round(x, 6) for x in conv2_w])});\n')
        f.write(f'export const CONV2_BIAS = new Float32Array({json.dumps([round(x, 6) for x in conv2_b])});\n')
        f.write(f'export const FC_WEIGHTS = new Float32Array({json.dumps([round(x, 6) for x in fc_w])});\n')
        f.write(f'export const FC_BIAS = new Float32Array({json.dumps([round(x, 6) for x in fc_b])});\n')

    print(f"\nWeights exported to: {output_path}")
    print(f"  Conv1: {len(conv1_w)} weights + {len(conv1_b)} bias")
    print(f"  Conv2: {len(conv2_w)} weights + {len(conv2_b)} bias")
    print(f"  FC:    {len(fc_w)} weights + {len(fc_b)} bias")
    print(f"  Total: {len(conv1_w)+len(conv1_b)+len(conv2_w)+len(conv2_b)+len(fc_w)+len(fc_b)} parameters")

if __name__ == '__main__':
    train()
