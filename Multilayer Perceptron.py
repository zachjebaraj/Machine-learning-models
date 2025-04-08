import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context
# OR, for safer practice:
# ssl._create_default_https_context = lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(certifi.where())


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten image
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nAccuracy on test set: {100 * correct / total:.2f}%")

# Visualize predictions
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data = example_data.to(device)
output = model(example_data)

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i].cpu().squeeze(), cmap='gray')
    _, pred = torch.max(output[i], 0)
    plt.title(f"Prediction: {pred.item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
