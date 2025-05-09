# Predicting Bike Demand Without Knowing the Equation (2-Hour Deep Learning Workshop)

Here, we will use a deep neural network to predict **bicycle rental demand** based on real-world inspired features. We’ll explore all steps in detail—from data simulation and preprocessing to model design, training, evaluation, and interpretation.

---

## 🚲 Objective

Predict the number of bikes rented on a given day using:

* Temperature (°C)
* Humidity (%)
* Wind speed (km/h)
* Day of the week (0=Sunday to 6=Saturday)
* Hour of the day (0-23)

We will **assume no known formula** for the relationship between inputs and output (demand), just like in real-world datasets. We'll use deep learning to learn this mapping from the data itself.

---

## 📦 Step 1: Simulate a Realistic Dataset

### 🧭 Supervised Learning Setup

In supervised learning, we train a model using examples that include both:

* **Inputs** (also called features): information we know and give to the model.
* **Targets** (also called labels): the outcome we want the model to learn to predict.

In our case:

* Inputs: `temperature`, `humidity`, `wind`, `day_of_week`, `hour`
* Target: `demand` (number of bikes rented)

We simulate these as if they came from historical records — no equation is used or known.

### 🔄 End-to-End Learning Flow

| Stage      | Input                      | Output                          | Role                        |
| ---------- | -------------------------- | ------------------------------- | --------------------------- |
| Training   | Inputs + Demand            | Learns weights to minimize loss | Supervised learning         |
| Testing    | Inputs only                | Predicts demand                 | Generalization / Evaluation |
| Evaluation | Predictions vs True Demand | MAE, RMSE, R² metrics           | Measure model performance   |

### 🧾 Sample Training Data (first 5 rows)

| temperature | humidity | wind  | day\_of\_week | hour | demand |
| ----------- | -------- | ----- | ------------- | ---- | ------ |
| 21.36       | 67.53    | 18.77 | 1             | 9    | 275.44 |
| 30.51       | 72.19    | 23.85 | 0             | 15   | 294.31 |
| 15.96       | 81.32    | 34.26 | 5             | 8    | 265.88 |
| 33.14       | 37.80    | 8.23  | 3             | 19   | 331.92 |
| 11.44       | 85.52    | 10.14 | 2             | 6    | 251.04 |
| 21.36       | 67.53    | 18.77 | 1             | 9    |        |
| 30.51       | 72.19    | 23.85 | 0             | 15   |        |
| 15.96       | 81.32    | 34.26 | 5             | 8    |        |
| 33.14       | 37.80    | 8.23  | 3             | 19   |        |
| 11.44       | 85.52    | 10.14 | 2             | 6    |        |

### 🧾 Sample Test Input (features only)

| temperature | humidity | wind  | day\_of\_week | hour |            |        |                |        |
| ----------- | -------- | ----- | ------------- | ---- | ---------- | ------ | -------------- | ------ |
| 25.62       | 60.20    | 12.40 | 6             | 10   |            |        |                |        |
| 18.37       | 70.12    | 5.50  | 4             | 17   |            |        |                |        |
| 29.88       | 50.44    | 14.80 | 1             | 13   | ---------- | ------ | -------------- | ------ |
| 25.62       | 60.20    | 12.40 | 6             | 10   |            |        |                |        |
| 18.37       | 70.12    | 5.50  | 4             | 17   |            |        |                |        |
| 29.88       | 50.44    | 14.80 | 1             | 13   |            |        |                |        |

These rows represent example records used in the training and testing process:

* **Training data** includes both features and the `demand` column so that the model can learn from examples.
* **Test data** includes only features — the model predicts `demand`, and we compare with the actual `demand` values (held back) to measure performance.
  . The goal is for the model to predict the `demand` column using these features — **without knowing any equation or rule upfront.**

### 💬 Code Explanation

#### 🧮 What If We Don't Know the Equation?

In real-world problems, we usually **don't know the equation** that links input features (like temperature, time, etc.) to the target output (like demand). That's where **deep learning** comes in.

Neural networks are powerful tools that can learn these complex, hidden relationships directly from the data — without us needing to explicitly define the mathematical formula.

So, in this tutorial, we act as though we do not know the equation. We simply provide:

* Historical input features: temperature, humidity, wind, time, day
* Observed output values: demand

The neural network then **learns the mapping** from inputs to outputs during training — by adjusting its weights and biases to reduce the prediction error.

> 📌 The goal of this lab is to simulate a real-world workflow where we rely entirely on data and use deep learning to uncover the patterns that drive predictions.

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

# Simulated observed demand values — in a real-world case, this would come from historical records
# No equation is assumed or known. These targets are treated as unknown relationships the model will learn.
demand = np.random.normal(loc=300, scale=50, size=rows)

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

### 🧾 Sample Prediction Table

| temperature | humidity | wind  | day\_of\_week | hour | True Demand | Predicted Demand |
| ----------- | -------- | ----- | ------------- | ---- | ----------- | ---------------- |
| 25.62       | 60.20    | 12.40 | 6             | 10   | 282.41      | 287.10           |
| 18.37       | 70.12    | 5.50  | 4             | 17   | 235.79      | 231.52           |
| 29.88       | 50.44    | 14.80 | 1             | 13   | 310.22      | 304.67           |

This table shows how closely the deep learning model was able to estimate demand based on input features — without knowing any mathematical equation.

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

