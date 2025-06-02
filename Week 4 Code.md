```python
# Install required packages (uncomment if running in Colab)
# !pip install torch torchvision scikit-learn matplotlib seaborn

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the digits dataset (8x8 grayscale images, 64 features per sample)
digits = load_digits()
X = digits.data            # Feature matrix (1797 samples x 64 features)
y = digits.target          # Labels (0 to 9)

# Normalize the input features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch dataset and dataloaders for batch training
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the MLP (Multilayer Perceptron) model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),     # Input layer to hidden layer 1
            nn.ReLU(),              # Activation for non-linearity
            nn.Linear(128, 64),     # Hidden layer 1 to hidden layer 2
            nn.ReLU(),              # Activation
            nn.Linear(64, 10)       # Output layer for 10 classes (digits 0–9)
        )
    
    def forward(self, x):
        return self.model(x)        # Forward pass

# Instantiate model, define loss function and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()  # Includes softmax inside
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
train_losses = []     # To track training loss
test_accuracies = []  # To track accuracy on test data

for epoch in range(epochs):
    model.train()     # Set model to training mode
    running_loss = 0.0

    # Iterate through training data
    for inputs, labels in train_loader:
        optimizer.zero_grad()            # Clear previous gradients
        outputs = model(inputs)          # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                  # Backward pass
        optimizer.step()                 # Update weights
        running_loss += loss.item()      # Accumulate loss

    # Average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate model on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():  # No need to compute gradients
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Show final classification report
print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds))

# Plot training loss and test accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker='o', color='green')
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

# Plot confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualize sample predictions from test set
def show_sample_predictions():
    plt.figure(figsize=(10, 4))
    for i in range(10):
        img = X_test[i].reshape(8, 8)  # Reshape to 8x8 image
        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {y_test[i]}\nPred: {all_preds[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_predictions()

```


---

```python
###
# Show a few sample images from the original dataset
def show_sample_input_images():
    plt.figure(figsize=(10, 4))
    for i in range(10):
        img = digits.images[i]  # Get 8x8 image
        label = digits.target[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("Sample Raw Input Images", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

show_sample_input_images()


# Show test images with model predictions
def show_test_input_output():
    model.eval()
    plt.figure(figsize=(12, 5))

    with torch.no_grad():
        for i in range(10):
            img = X_test_tensor[i].reshape(1, -1)
            output = model(img)
            _, pred = torch.max(output, 1)

            original_image = X_test[i].reshape(8, 8)
            plt.subplot(2, 5, i + 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f"True: {y_test[i]}\nPred: {pred.item()}")
            plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Predictions on Test Set", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

show_test_input_output()




```
