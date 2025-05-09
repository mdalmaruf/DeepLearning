# 🏠 Case Study: Predict House Prices Using a Simple Neural Network (with PyTorch)

This tutorial introduces students to building, training, and deeply understanding a **basic neural network** for **house price prediction** using **PyTorch**. We’ll not only implement the code but also **explain every step mathematically** and conceptually — ideal for a 1-hour class.

---

## 🎯 Learning Objectives

By the end of this session, you will:

* Understand how a basic neural network models linear relationships
* Learn what a neuron is and how it processes input
* Break down the concepts of weight, bias, activation, loss, and gradient
* See how forward and backward propagation works mathematically
* Train a model step-by-step with real examples

---

## 🧠 What Is a Neural Network (Model)?

A neural network is a **function approximator**. At its simplest, it looks like this:

$$
\hat{y} = w \cdot x + b
$$

This is **exactly the same** as linear regression, where:

* **x** = input (e.g., house size)
* **w** = weight (how much input affects output)
* **b** = bias (base value when x = 0)
* **\hat{y}** = predicted output (e.g., price)

### 📦 What Are We Fitting?

We are trying to learn the best **weight and bias** that minimizes the difference between our predicted price and the actual price.

Our model starts with random values for weight and bias. As we see more data and calculate **error**, we adjust the weight and bias using **gradient descent** to make better predictions.

---

## 📅 Step 1: Setup and Import Libraries

```python
!pip install torch
import torch
import torch.nn as nn
```

---

## 📊 Step 2: Define a Simple House Price Dataset

```python
# Inputs (scaled: 1000 sqft = 1.0)
x = torch.tensor([[1.0], [1.5], [2.0]])

# Outputs (in $1000s)
y = torch.tensor([[300.0], [450.0], [500.0]])
```

---

## 🧱 Step 3: Define the Model (1 Neuron)

```python
model = nn.Linear(in_features=1, out_features=1)
```

This model learns:

$$
\hat{y} = w \cdot x + b
$$

Check initial values:

```python
print("Initial weight:", model.weight.item())
print("Initial bias:", model.bias.item())
```

---

## 🔢 Step 4: Manually Predict with a Forward Pass

Let’s assume:

* w = 100
* b = 200
* x = 1.0 (1000 sqft)

Then:

$$
\hat{y} = 100 \cdot 1 + 200 = 300 \text{ (perfect match!)}
$$

Try this using PyTorch:

```python
with torch.no_grad():
    y_pred = model(x)
    print("Predicted Prices:", y_pred.squeeze())
```

---

## ⚖️ Step 5: Define Loss Function

Mean Squared Error (MSE):

$$
L = \frac{1}{n} \sum (\hat{y} - y)^2
$$

```python
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y)
print("Initial Loss:", loss.item())
```

This tells us how far off we are on average.

---

## 🔙 Step 6: Backpropagation - Calculate Gradients

```python
model.zero_grad()
loss.backward()
```

Now inspect the gradients:

```python
print("Gradient (∂L/∂w):", model.weight.grad.item())
print("Gradient (∂L/∂b):", model.bias.grad.item())
```

These tell us **how much** changing the weight/bias would affect the loss.

---

## 🔄 Step 7: Update Weights Manually

Using gradient descent:

$$
w := w - \eta \cdot \frac{\partial L}{\partial w}
$$

Where $\eta = 0.01$ (learning rate).

```python
learning_rate = 0.01

with torch.no_grad():
    model.weight -= learning_rate * model.weight.grad
    model.bias -= learning_rate * model.bias.grad
```

Then predict again:

```python
new_preds = model(x)
new_loss = loss_fn(new_preds, y)
print("Updated Loss:", new_loss.item())
```

---

## 🧮 Deep Dive: Full Math Example

Suppose:

* x = 1.5 (1500 sqft)
* y = 450
* Initial w = 100, b = 100

### Forward pass:

$$
\hat{y} = 100 \cdot 1.5 + 100 = 250
$$

Loss:

$$
L = (\hat{y} - y)^2 = (250 - 450)^2 = 40000
$$

### Gradients:

* $\frac{\partial L}{\partial \hat{y}} = 2 \cdot (250 - 450) = -400$
* $\frac{\partial L}{\partial w} = -400 \cdot 1.5 = -600$
* $\frac{\partial L}{\partial b} = -400$

### Update:

* $w = 100 + 6 = 106$
* $b = 100 + 4 = 104$

### New prediction:

$$
\hat{y} = 106 \cdot 1.5 + 104 = 263 + 104 = 367
$$

Closer to 450! Keep updating to improve.

---

## 🔁 Full Training Loop (with Optimizer)

```python
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(10):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: weight = {model.weight.item():.2f}, bias = {model.bias.item():.2f}, loss = {loss.item():.2f}")
```

---

## 📚 Key Takeaways

* **Weight** and **bias** are learned through loss feedback
* The **forward pass** computes predictions
* The **loss function** quantifies error
* **Backpropagation** computes gradients
* **Gradient descent** updates the parameters to minimize error

---

Next time: We'll add **hidden layers** and **activation functions** to model non-linear relationships.
