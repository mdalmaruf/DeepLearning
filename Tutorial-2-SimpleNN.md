## Case Study: Predicting Car Headlight Brightness Based on Ambient Light

### Problem Scenario

Modern cars use sensors to **automatically adjust headlight brightness** depending on how bright or dark it is outside.

You're asked to build a **simple neural network** that mimics this behavior:

* In **bright daylight**, the headlights are dim or off.
* In **dark environments**, the headlights are at full brightness.
* The transition should be **smooth and adaptive**, not on/off.

---

### 🌟 Goal

Design a neural network that:

* Takes **ambient light level** (0–100) as input
* Outputs a **brightness value** (0–1), where 1 = max brightness

This problem has an **inverse nonlinear relationship**:

$$
\text{More ambient light} \Rightarrow \text{Less headlight brightness}
$$

---

## 🧠 Network Design

| Layer        | Details                                      |
| ------------ | -------------------------------------------- |
| Input        | 1 feature (Ambient light level: 0 to 100)    |
| Hidden Layer | 3 neurons with ReLU activation               |
| Output       | 1 neuron for headlight brightness prediction |

---

## Code: Neural Network with Predefined Weights

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the model
class HeadlightControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)

        # Manually set weights and biases to simulate behavior
        self.layer1.weight = nn.Parameter(torch.tensor([[-0.1], [-0.05], [-0.2]]))
        self.layer1.bias = nn.Parameter(torch.tensor([10.0, 5.0, 15.0]))
        self.layer2.weight = nn.Parameter(torch.tensor([[0.6, 0.3, 0.5]]))
        self.layer2.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

---

### Visualize Headlight Brightness Based on Ambient Light

```python
# Instantiate model and generate predictions for ambient light levels
model = HeadlightControlNet()
ambient_light = torch.linspace(0, 100, 100).reshape(-1, 1)
brightness = model(ambient_light).detach()

# Plot results
plt.plot(ambient_light.numpy(), brightness.numpy(), label="Headlight Brightness")
plt.xlabel("Ambient Light Level (0 = dark, 100 = bright)")
plt.ylabel("Headlight Brightness (0–1)")
plt.title("Car Headlight Control Using Neural Network")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Learned

* How to simulate an **inverse nonlinear function** using neural networks
* How **ReLU activation** enables conditional behavior (on only when needed)
* The idea that **weights control direction** and **biases shift activation**

---

## Student Tasks

1. Modify weights and biases to adjust how quickly brightness fades with light.
2. Replace ReLU with `Sigmoid` or `Tanh` and see how that changes the transition.
3. Add a second hidden layer and observe whether the shape becomes smoother.
4. Can you make it behave more like an actual car — e.g., full brightness until ambient light > 30?
