```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Step 1: Prepare the dataset
# ---------------------------
sequence = "machine"
chars = sorted(list(set(sequence)))  # Unique sorted characters
char_to_idx = {ch: i for i, ch in enumerate(chars)}     # Mapping char -> index
idx_to_char = {i: ch for ch, i in char_to_idx.items()}  # Mapping index -> char
vocab_size = len(chars)

# Convert characters to indices
input_seq = [char_to_idx[ch] for ch in sequence[:-1]]   # "m a c h i n"
target_seq = [char_to_idx[ch] for ch in sequence[1:]]   # "a c h i n e"

# Convert to tensors of shape (sequence_length, batch_size)
input_tensor = torch.tensor(input_seq).unsqueeze(1)
target_tensor = torch.tensor(target_seq).unsqueeze(1)

# ---------------------------
# Step 2: Define the RNN model
# ---------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)     # Embedding layer
        self.rnn = nn.RNN(hidden_size, hidden_size)            # Vanilla RNN layer
        self.fc = nn.Linear(hidden_size, vocab_size)           # Output projection

    def forward(self, input_seq, hidden):
        embedded = self.embed(input_seq)                       # Embed input
        output, hidden = self.rnn(embedded, hidden)            # Apply RNN
        logits = self.fc(output)                               # Project to vocab space
        return logits, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)             # Initial hidden state

# ---------------------------
# Step 3: Train the model
# ---------------------------
hidden_size = 16
model = CharRNN(vocab_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 300
losses = []
all_predictions = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    hidden = model.init_hidden()                       # Reset hidden state each epoch
    optimizer.zero_grad()

    output, hidden = model(input_tensor, hidden)       # Forward pass
    loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))  # Loss
    loss.backward()                                    # Backward pass
    optimizer.step()                                   # Update weights

    losses.append(loss.item())                         # Store loss

    if (epoch + 1) % 50 == 0:
        pred_idx = output.argmax(dim=2).squeeze().tolist()            # Get predicted indices
        pred_str = ''.join([idx_to_char[i] for i in pred_idx])        # Convert to string
        all_predictions.append((epoch + 1, loss.item(), pred_str))    # Store predictions
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Prediction: {pred_str}")

# ---------------------------
# Step 4: Inference Function
# ---------------------------
def predict(model, start_char, length=6):
    model.eval()
    input_char = torch.tensor([char_to_idx[start_char]]).unsqueeze(1)
    hidden = model.init_hidden()
    output_seq = start_char

    for _ in range(length):
        output, hidden = model(input_char, hidden)                  # Forward pass
        top_idx = output.argmax(dim=2)[-1].item()                   # Pick most probable char
        next_char = idx_to_char[top_idx]
        output_seq += next_char                                     # Append to output
        input_char = torch.tensor([[top_idx]])                      # Prepare next input

    return output_seq

# ---------------------------
# Step 5: Evaluation
# ---------------------------
final_pred_idx = output.argmax(dim=2).squeeze().tolist()
final_pred_str = ''.join([idx_to_char[i] for i in final_pred_idx])
target_str = ''.join([idx_to_char[i] for i in target_tensor.squeeze().tolist()])
correct_chars = sum(p == t for p, t in zip(final_pred_str, target_str))
accuracy = correct_chars / len(target_str)

print("\n=== Evaluation ===")
print("Target       :", target_str)
print("Prediction   :", final_pred_str)
print(f"Accuracy     : {accuracy:.2%}")
print("Inference    :", predict(model, 'm'))

# ---------------------------
# Step 6: Plot training loss
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rnn_training_loss_plot.png")
plt.show()

```
