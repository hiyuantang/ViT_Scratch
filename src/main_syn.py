import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchinfo
from utils import *
from VisionTransformer import RegressionViT
from SyntheticDataset import SyntheticDataset

# Step 1: Load Synthetic dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SyntheticDataset(data_dir='E:\db_synthetic_1', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# Train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Example Training
image_size = 224
patch_size = 16
in_channels = 3  # RGB channels
embed_dim = 256

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
batch_size = 64
epochs = 50

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = RegressionViT(patch_size=patch_size, num_patches=196, num_blocks=8, 
                        embed_dim=embed_dim, hidden_dim=256, num_heads=32, dropout=0.1).to(device)

torchinfo.summary(model, input_size=(32, 196, 3, 16, 16))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
best_val_mse = float('inf')
early_stopping_patience = 10  # Number of epochs to wait before early stopping
early_stopping_counter = 0
best_model_state_dict = None

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Send data and targets to device
        data = data.to(device)
        data = Patching_CLS(data, patch_size=patch_size, device=device)
        targets = targets.view(-1, 1).to(device)

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
        mse_loss = 0
        num_samples = 0
        for data, targets in val_loader:
            # Send data to device
            data = data.to(device)
            data = Patching_CLS(data, patch_size=patch_size, device=device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)

            # Calculate MSE loss
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(scores, targets.unsqueeze(1))  # Unsqueeze targets to match scores shape

            mse_loss += loss.item() * data.size(0)
            num_samples += data.size(0)

        mse_loss /= num_samples
        print(f'Validation MSE loss after epoch {epoch+1}: {mse_loss:.4f}')

        # Save the model with the best validation MSE
        if mse_loss < best_val_mse:
            best_val_mse = mse_loss
            best_model_state_dict = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping after {epoch+1} epochs.')
            break

print("Training complete")

# Save the best model
torch.save(best_model_state_dict, 'best_model.pth')

# Load the best model
best_model = RegressionViT(patch_size=patch_size, num_patches=196, num_blocks=8, 
                        embed_dim=embed_dim, hidden_dim=256, num_heads=32, dropout=0.1).to(device)
best_model.load_state_dict(torch.load('best_model.pth'))

true_labels = []
estimations = []

with torch.no_grad():
    for data, targets in data_loader:
        data = data.to(device)
        data = Patching_CLS(data, patch_size=patch_size, device=device)
        targets = targets.to(device)

        scores = best_model(data)

        true_labels.extend(targets.cpu().numpy())
        estimations.extend(scores.squeeze().cpu().numpy())

# Plot the true labels vs. estimations
import matplotlib.pyplot as plt
# Plot the true labels vs. estimations
plt.figure(figsize=(8, 6))
plt.scatter(true_labels, estimations, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Estimations')
plt.title(f'True Labels vs. Estimations (MSE = {best_val_mse:.4f})')

# Add a line where y = x (perfect fit)
min_val = min(min(true_labels), min(estimations))
max_val = max(max(true_labels), max(estimations))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
plt.legend()
plt.show()