
# Introduction to Neural Networks and Deep Learning
## Week 1 - Build Your First Neural Network in PyTorch

Welcome to the first lecture in our Deep Learning course. In this session, we will introduce the concept of **neural networks**, how they differ from traditional models like linear regression, and walk you through your first real neural network implementation using **PyTorch** in **Google Colab**.

---

## What is a Neural Network?

A **neural network** is a set of algorithms inspired by the human brain, designed to recognize patterns. It is the foundation of modern **deep learning**, and it can be used for:

- Image and speech recognition
- Natural language processing
- Predictive analytics
- Game playing (e.g., AlphaGo)

---

## Basic Structure of a Neural Network

A basic neural network includes:
- **Input Layer**: Accepts raw data (e.g., pixel values, tabular features).
- **Hidden Layers**: Where computations happen using **weights**, **biases**, and **activation functions**.
- **Output Layer**: Produces predictions.

Each **neuron** computes:
\[
z = w \cdot x + b,\quad \text{then passes through activation like ReLU: } a = \max(0, z)
\]

### Neural Network Flow (Visual Example):

![Neural Net Diagram](https://raw.githubusercontent.com/StatQuest/signa/main/chapter_01/images/chapter_1_pre_trained_nn.png)

And here is another simple diagram (StatQuest style):

![Dose-to-Effectiveness Flow](attachment:52d02905-4059-4650-a283-96c2b51b351e.png)

---

## Why Neural Networks Are Powerful

Unlike linear models, neural networks:
- Can model **nonlinear** relationships
- Use **activation functions** to bend and shape data
- Are **composable** — we can stack multiple layers to form a **deep network**

---

## 🔧 Setting Up PyTorch in Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new Python 3 notebook
3. Run the following command to install PyTorch:
```python
!pip install torch torchvision matplotlib
```

---

## Your First Demo: A Simple 3-Layer Neural Network

### Task:
We will use a **pre-trained neural network** (weights and biases are already given) to understand how inputs flow through the network and how each layer transforms the data.

### Step-by-step Guide

#### 1. Import Required Libraries
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

#### 2. Create the Neural Network Class with Pre-trained Weights
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)

        # Manually setting weights and biases (pretend they were trained)
        self.layer1.weight = nn.Parameter(torch.tensor([[2.0], [-3.0], [0.5]]))
        self.layer1.bias = nn.Parameter(torch.tensor([0.5, 0.1, -0.3]))
        self.layer2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
        self.layer2.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

#### 3. Run a Forward Pass
```python
net = SimpleNet()
x_vals = torch.linspace(-2, 2, 100).reshape(-1, 1)
y_vals = net(x_vals).detach()

plt.plot(x_vals.numpy(), y_vals.numpy(), label="Neural Net Output")
plt.title("First Neural Network Demo")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Learned

- How to define a neural network in PyTorch using `nn.Module`
- What a forward pass means (input → layer → output)
- How activation functions like ReLU shape the output
- Visualizing the **nonlinear transformation** that a neural network learns

---

## Key Terms Recap

| Term              | Meaning                                                                 |
|-------------------|-------------------------------------------------------------------------|
| **Weight**         | Multiplier for each input feature                                       |
| **Bias**           | Offset added after multiplication                                       |
| **Activation**     | Nonlinear function like ReLU or Sigmoid                                 |
| **Forward pass**   | Feeding input through the network to get predictions                    |
| **Layer**          | A group of neurons that perform transformations                         |

---

## Suggested Homework

1. Try changing the weights and biases in the `SimpleNet` class.
2. Add a second hidden layer and observe changes in the output.
3. Research what `ReLU` does compared to `Sigmoid`.

---

## 🧩 Coming Up Next...

In the next lecture, we will build a **trainable neural network** from scratch and use a real dataset to learn weights — introducing concepts like **loss functions** and **gradient descent**.

---
