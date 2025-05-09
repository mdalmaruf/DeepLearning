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

---

## 🧹 Step 2: Normalize and Convert to Tensors

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

# Important Note
In this lab, we define the demand equation ourselves to control patterns and test model behavior. In the real world, you wouldn’t know this equation — you’d let the model discover it from the data.
