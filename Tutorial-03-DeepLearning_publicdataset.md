# 🧪 Tutorial 4: Deep Learning Regression with Real-World Dataset – Wine Quality

This tutorial uses the UCI Wine Quality dataset to predict wine ratings based on physicochemical properties. It introduces key concepts like loss functions, learning rate, batch size, optimizers, activation functions, and neural network architecture.

---

## 🍷 Dataset: Wine Quality

### 🧾 Sample Data Table

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH   | sulphates | alcohol | quality |
| ------------- | ---------------- | ----------- | -------------- | --------- | ------------------- | -------------------- | ------- | ---- | --------- | ------- | ------- |
| 7.4           | 0.70             | 0.00        | 1.9            | 0.076     | 11                  | 34                   | 0.9978  | 3.51 | 0.56      | 9.4     | 5       |
| 7.8           | 0.88             | 0.00        | 2.6            | 0.098     | 25                  | 67                   | 0.9968  | 3.20 | 0.68      | 9.8     | 5       |
| 7.8           | 0.76             | 0.04        | 2.3            | 0.092     | 15                  | 54                   | 0.9970  | 3.26 | 0.65      | 9.8     | 5       |

We use the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality), which includes:

**Features:**

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

**Target:**

* Quality (score from 0 to 10)

---

## 📦 Step 1: Load and Preprocess the Data

This step loads the dataset, splits it into training and testing sets, and normalizes the feature values using `StandardScaler`. Normalization is crucial for neural networks to converge efficiently during training.

```python
# 📥 Import libraries for data loading and preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Separate input features and target column (quality score)
X = data.drop('quality', axis=1)
y = data['quality']

# Split into 80% training and 20% test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input features to zero mean and unit variance using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 🔧 Step 2: Build the Neural Network

Here we define a feedforward neural network using `nn.Sequential`. It contains two hidden layers with ReLU activations. This structure is simple, readable, and works well for tabular data.

```python
import torch
import torch.nn as nn

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
```

---

## ⚙️ Step 3: Configure Loss and Optimizer

MSE is a standard choice for regression problems. Adam is a powerful optimizer that adapts the learning rate for each parameter during training.

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Loss Function:** MSE (Mean Squared Error) is used for regression.

**Optimizer:** Adam adapts the learning rate for each parameter and generally performs well.

---

## 🏋️ Step 4: Train the Model

We use 100 epochs as a default for this dataset. However, the ideal number of epochs can vary depending on the complexity of the dataset and the model. To avoid overfitting or wasting resources, it's common to monitor the loss curve or implement early stopping.

We train the model for a fixed number of epochs (100). The data is shuffled and batched to ensure a better learning gradient. Each epoch includes multiple batch updates using backpropagation.

```python
# Set training parameters
# Consider using early stopping when validation loss starts increasing
# Example: start with 100 epochs and monitor performance visually

epochs = 100
batch_size = 32
losses = []

for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0

    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / (X_train_tensor.size()[0] / batch_size))
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
```

**Epoch**: One full pass through the training data.

**Batch Size**: Number of samples used to estimate the gradient before each update.

**Learning Rate**: Controls the step size during weight updates.

---

## 📈 Step 5: Evaluate the Model

After training, we evaluate the model using common regression metrics: MAE, RMSE, and R² score. We also visualize how well the predictions align with true values using a scatter plot.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Plot
plt.figure(figsize=(10,5))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.grid(True)
plt.show()

# 🔍 Visualize loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve During Training')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📚 Key Concepts Explained

### 🧾 Why We Chose These Settings

* **Hidden Layers (2 layers, 64 and 32 neurons)**: This architecture is simple yet expressive enough to model patterns in small-to-medium tabular datasets like wine quality. More layers could overfit the data, while fewer might underfit.
* **Epochs (100)**: This number gives the model enough time to learn while still being fast to train. If validation loss increases, it may be too much.
* **Batch Size (32)**: A moderate batch size balances gradient noise (which can help generalization) and computational efficiency.
* **Learning Rate (0.001)**: This is the default in Adam and works well in most cases. A higher rate might cause overshooting, while a lower one could slow down learning or get stuck.
* **ReLU Activation**: Chosen for its simplicity and speed. It avoids vanishing gradients common in sigmoid/tanh and is suitable for hidden layers in most regression and classification tasks.
* **Adam Optimizer**: Selected for its ability to adapt the learning rate and converge faster without requiring a lot of manual tuning.

📌 Changing any of these settings:

* Using too few neurons → underfitting.
* Too many → overfitting.
* Large batch size (e.g. 128) → smooth training but worse generalization.
* Very high learning rate → loss may diverge.
* Replacing ReLU with sigmoid → risk of vanishing gradients.
* Using SGD without momentum → might need careful learning rate tuning and converge slower.

Use visualizations like the **loss curve** and validation metrics to fine-tune these choices.

### 🔍 How Many Hidden Layers and Neurons?

* For structured/tabular data like this wine dataset, **1 to 2 hidden layers** is usually sufficient.
* **Too few neurons** may lead to underfitting. **Too many** can overfit.
* A good rule of thumb: start with a layer of 1–2× the number of input features.
* Here, we used 64 → 32 → 1:

  * 11 input features → 64 neurons (1st layer)
  * 32 neurons (2nd layer)
  * 1 neuron in the output (for regression)

### 🧠 When to Stop Training (Early Stopping)

* Monitor validation loss: stop when it stops improving.
* Tools like `EarlyStopping` from `torchkeras` or `pytorch_lightning` help.
* Also visualize the loss curve — plateau or increase in loss suggests stopping.

### 📏 What Does the StandardScaler Do?

* StandardScaler standardizes features to have **mean 0 and variance 1**:

  ```
  x' = (x - mean) / std
  ```
* This ensures that all features contribute equally and training converges faster.
* Without scaling, features with large ranges (e.g., alcohol %) could dominate others.

### 🔄 Why Use `nn.Sequential`?

* `nn.Sequential` is ideal when layers are stacked linearly (i.e., no branching or custom connections).
* It’s concise and readable for most feedforward networks.
* For more complex networks (e.g., shared layers, residuals), use a custom `nn.Module` class:

  ```python
  class WineModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.hidden = nn.Linear(11, 64)
          self.output = nn.Linear(64, 1)

      def forward(self, x):
          x = torch.relu(self.hidden(x))
          return self.output(x)
  ```

### 🧪 Which Activation Functions and When?

* **ReLU (Rectified Linear Unit)**: Default for most hidden layers. It outputs zero for negative inputs and passes through positive inputs. Helps mitigate vanishing gradient problems and is computationally efficient.

  * Example: Used in this lab between layers to introduce non-linearity.
* **Sigmoid**: Compresses output between 0 and 1. Good for binary classification outputs.

  * Use case: When output represents a probability (e.g., spam vs. not spam).
* **Tanh**: Similar to sigmoid but outputs between -1 and 1. Can help in models where zero-centered outputs are useful.

  * Use case: Shallow networks or some RNN applications.
* **Softmax**: Used in multi-class classification problems to convert raw outputs into probabilities summing to 1.

  * Use case: Final layer in a digit recognition model (0–9 classification).

👉 In regression tasks (like this one), we usually do **not** apply an activation function in the final layer, so the model can output any continuous value.

* **ReLU**: Default for hidden layers. Fast, works well, avoids vanishing gradients.
* **Sigmoid**: Use when output must be between 0 and 1 (e.g., binary classification).
* **Tanh**: Similar to sigmoid but zero-centered. Can be useful in shallow networks.
* **Softmax**: Use in the final layer for **multi-class classification** problems.

### 🧮 Epochs and Batch Size — Impact

* **Epochs**: Number of times the full dataset is passed through the model.

  * Too low = underfitting. Too high = overfitting.
  * Typical range: 50–500 for small/medium datasets.
* **Batch Size**:

  * Small (16–32): More noise, more updates, good generalization.
  * Large (64–128): Smoother gradients, faster on GPU, but may overfit or converge to sharp minima.

These concepts help design better models and tune them for optimal performance.

### Activation Functions

* **ReLU**: Most commonly used in hidden layers.
* **Sigmoid/Tanh**: Better for outputs in (0,1) or (-1,1) — not ideal for deep hidden layers.

### Choosing Hidden Layers & Neurons

* **Start small** (1–2 layers, 16–64 neurons)
* **Add depth** for complexity (e.g., image processing)

### Optimizers

| Optimizer   | Description                                                                  | When to Use                                                      |
| ----------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **SGD**     | Vanilla stochastic gradient descent. Simple, may require tuning.             | Small or interpretable models, good when training data is small. |
| **Adam**    | Adaptive learning rate with momentum. Works well out of the box.             | Recommended default for most deep learning models.               |
| **RMSprop** | Designed for RNNs. Scales learning rate based on recent gradient magnitudes. | Time-series or sequential data.                                  |

* **SGD**: Simple, fast, may need manual learning rate tuning
* **Adam**: Generally better performance with minimal tuning

### Learning Rate

* Too high: may overshoot the minimum
* Too low: slow convergence

### Epochs and Batch Size

* **More epochs** → better learning but risk of overfitting
* **Smaller batch** → more updates per epoch, noisier gradients

---

This lab walks students through using deep learning for a real-world regression task and builds strong intuition about model tuning, architecture, and evaluation.
