import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchinfo
from utils import *
from VisionTransformer import ClassificationViT

# Step 1: Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to desired size
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for each channel
        std=[0.229, 0.224, 0.225]    # Standard deviation for each channel
    )
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Train-test split
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# selected_categories = [0, 1]

# # Create custom subsets with only the selected categories
# train_subset = torch.utils.data.Subset(train_dataset, indices=[i for i, label in enumerate(train_dataset.targets) if label in selected_categories])
# test_subset = torch.utils.data.Subset(test_dataset, indices=[i for i, label in enumerate(test_dataset.targets) if label in selected_categories])

# # Train-test split
# train_size = int(0.8 * len(train_subset))
# test_size = len(train_subset) - train_size
# train_subset, val_subset = torch.utils.data.random_split(train_subset, [train_size, test_size])

# Example Training
image_size = 32
patch_size = 8
in_channels = 3  # RGB channels
embed_dim = 128

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
batch_size = 64
epochs = 20

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = ClassificationViT(num_class=10, patch_size=patch_size, num_patches=16, num_blocks=8, 
                          embed_dim=embed_dim, hidden_dim=128, num_heads=32, dropout=0.1).to(device)

torchinfo.summary(model, input_size=(64, 16, 3, 8, 8))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Send data and targets to device
        data = data.to(device)
        data = Patching_CLS(data, patch_size=patch_size, device=device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validation loop
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for data, targets in val_loader:
            # Send data to device
            data = data.to(device)
            data = Patching_CLS(data, patch_size=patch_size, device=device)
            targets = targets.to(device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        val_accuracy = float(num_correct) / float(num_samples)
        print(f'Validation accuracy after epoch {epoch+1}: {val_accuracy * 100:.2f}%')

print("Training complete")