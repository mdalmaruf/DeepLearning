# Nonlinear Regression on Bicycle Rental Demand

In this hands-on tutorial, we will use a deep neural network to predict **bicycle rental demand** based on real-world inspired features. We’ll explore all steps in detail—from data simulation and preprocessing to model design, training, evaluation, and interpretation.

---

## 🚲 Objective

Predict the number of bikes rented on a given day using:

* Temperature (°C)
* Humidity (%)
* Wind speed (km/h)
* Day of the week (0=Sunday to 6=Saturday)
* Hour of the day (0-23)

We’ll simulate data with nonlinear trends and interactions to learn how a deep learning model can fit complex relationships.

---

## 📦 Step 1: Simulate a Realistic Dataset

### 💬 Code Explanation

* `np.random.seed(0)` ensures reproducibility — you get the same data every time you run the code.
* `rows = 500` means we are simulating 500 rows of data.
* **Feature simulation**:

  * `temperature`: values between 5°C and 35°C.
  * `humidity`: values between 30% and 90%.
  * `wind`: wind speed in km/h between 0 and 40.
  * `day_of_week`: integers from 0 to 6, where 0 is Sunday.
  * `hour`: integers from 0 to 23, representing hour of the day.
* **Target (`demand`)** is computed using a nonlinear equation that models how real-world bike rental demand works:

  * Increases with temperature
  * Decreases with humidity and wind
  * Peaks during morning and evening hours due to cosine function
  * Is higher on weekends (Saturday and Sunday)
  * Includes Gaussian noise to simulate measurement errors or randomness
* `pd.DataFrame(...)` constructs a pandas DataFrame to organize the simulated dataset for easy processing and viewing.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(0)
rows = 500

# Simulated features
temperature = np.random.uniform(5, 35, rows)  # in Celsius
humidity = np.random.uniform(30, 90, rows)    # in %
wind = np.random.uniform(0, 40, rows)         # in km/h
day_of_week = np.random.randint(0, 7, rows)   # 0=Sunday, ..., 6=Saturday
hour = np.random.randint(0, 24, rows)

# Simulated target (nonlinear with noise)
demand = (
    100 + 3 * temperature 
    - 2 * humidity 
    - 0.5 * wind 
    + 20 * np.cos(np.pi * hour / 12)  # morning/evening peak
    + 10 * (day_of_week >= 5)         # weekend boost
    + np.random.normal(0, 10, rows)   # noise
)

# Construct DataFrame
bike_df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind': wind,
    'day_of_week': day_of_week,
    'hour': hour,
    'demand': demand
})
```
$$demand = 100 + 3T - 2H - 0.5W + 20 * cos(πh / 12) + 10 * I\_weekend + ε$$
---

## 🧹 Step 2: Normalize and Convert to Tensors

### 💬 Code Explanation

* We first split the dataset into **input features** (`temperature`, `humidity`, etc.) and the **target** (`demand`).
* `MinMaxScaler()` rescales values to range between 0 and 1 — this helps neural networks converge faster.
* `train_test_split(...)` divides the dataset into 80% training and 20% testing.
* `torch.tensor(..., dtype=torch.float32)` converts NumPy arrays into PyTorch tensors, which are required for training the model using PyTorch operations.

```python
from sklearn.model_selection import train_test_split

features = bike_df[['temperature', 'humidity', 'wind', 'day_of_week', 'hour']].values
target = bike_df[['demand']].values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```

---

## 🏗️ Step 3: Define a Deep Neural Network

### 💬 Code Explanation

* `nn.Sequential(...)` allows us to stack layers easily in PyTorch.
* `nn.Linear(5, 32)` means the input layer has 5 features, and the first hidden layer has 32 neurons.
* `nn.ReLU()` introduces non-linearity to the model — without this, the network would only learn linear functions.
* We use two hidden layers of size 32 with ReLU, followed by an output layer `nn.Linear(32, 1)` that produces one continuous output: the predicted bike demand.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
```

---

## ⚙️ Step 4: Compile and Train the Model

### 💬 Code Explanation

* `optimizer = torch.optim.Adam(...)` uses the Adam optimizer, which adapts the learning rate during training and usually performs better than plain SGD.
* `nn.MSELoss()` is the loss function — Mean Squared Error is appropriate for regression tasks.
* We loop through 300 **epochs** (complete passes through the training data):

  * `model.train()` sets the model to training mode.
  * `pred = model(x_train)` runs a forward pass.
  * `loss.backward()` computes gradients for each parameter.
  * `optimizer.step()` updates the model parameters.
  * Every 50 epochs, we print the loss so students can track progress.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(300):
    model.train()
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## 📈 Step 5: Plot Loss Curve

### 💬 Code Explanation

* After training, we store the loss values in a list called `losses`.
* This code block plots the **loss curve**, showing how the error reduces over time.
* It helps verify if the model is learning effectively. A decreasing trend indicates improvement.
* If the curve flattens or increases, it could signal overfitting or a learning rate issue.

```python
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
```

---

## 🧪 Step 6: Evaluate the Model

### 💬 Code Explanation

* We evaluate the model on **test data** to see how well it generalizes.
* `model.eval()` switches to evaluation mode (e.g., disables dropout if any).
* `torch.no_grad()` ensures gradients aren't tracked during prediction.
* We **inverse transform** the predictions and labels to their original scales for proper interpretation.
* **Evaluation metrics**:

  * `MAE`: Average absolute difference between prediction and true values
  * `RMSE`: Square root of mean squared error (penalizes large errors)
  * `R²`: Measures how well predictions explain the variance in data (closer to 1 is better)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model.eval()
with torch.no_grad():
    y_pred = model(x_test).numpy()
    y_true = y_test.numpy()

    y_pred = scaler_y.inverse_transform(y_pred)
    y_true = scaler_y.inverse_transform(y_true)

print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
print("R²:", r2_score(y_true, y_pred))
```

---

## 📊 Step 7: Visualize Predictions vs Actual

### 💬 Code Explanation

* This plot overlays the model's predictions with the actual demand values from the test set.
* Helps visually inspect where predictions are close or far off.
* Ideal outcome: predicted line should track close to the actual line.
* Discrepancies can highlight underfitting or noisy inputs.

```python
plt.figure(figsize=(10, 5))
plt.plot(y_true, label='True Demand')
plt.plot(y_pred, label='Predicted Demand')
plt.legend()
plt.title("Predicted vs True Bike Demand")
plt.xlabel("Sample")
plt.ylabel("Number of Bikes Rented")
plt.grid(True)
plt.show()
```

---

## 🧠 Key Learning Outcomes

* Understand how to simulate realistic nonlinear datasets
* Learn deep regression with PyTorch using `nn.Sequential`
* Observe how deeper layers improve prediction accuracy
* Evaluate with MAE, RMSE, and R²

✅ Encourage students to modify features (e.g., add `is_holiday`, `rain`, etc.) and re-train to see the model behavior.


# Important Note
In this lab, we define the demand equation ourselves to control patterns and test model behavior. In the real world, you wouldn’t know this equation — you’d let the model discover it from the data.
