import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Get user input
text = input("Enter a training string (e.g., 'banana', 'hello world'): ").lower()
if len(text) < 6:
    raise ValueError("Please enter at least 6 characters.")

model_type = input("Choose model type: [LSTM / GRU] ").strip().lower()
gen_len = int(input("How many characters to generate after training? (e.g., 20): ").strip())

# Step 2: Preprocess text
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Create sequences
seq_length = 5
x_data, y_data = [], []
for i in range(len(text) - seq_length):
    x_data.append([char2idx[ch] for ch in text[i:i+seq_length]])
    y_data.append([char2idx[ch] for ch in text[i+1:i+seq_length+1]])

x_tensor = torch.tensor(x_data).long()
y_tensor = torch.tensor(y_data).long()

# Step 3: Define model (LSTM or GRU)
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, use_gru=False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.rnn = nn.GRU(vocab_size, hidden_size, batch_first=True) if use_gru else nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

use_gru = model_type == "gru"
model = RNNModel(vocab_size=vocab_size, hidden_size=64, use_gru=use_gru)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 4: Training loop
for epoch in range(100):
    output = model(x_tensor)
    loss = criterion(output.view(-1, vocab_size), y_tensor.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

# Step 5: Generate text
model.eval()
seed = text[:seq_length]
generated = seed

input_seq = torch.tensor([[char2idx[ch] for ch in seed]]).long()
with torch.no_grad():
    for _ in range(gen_len):
        output = model(input_seq)
        pred = torch.argmax(output[:, -1, :], dim=-1).item()
        next_char = idx2char[pred]
        generated += next_char

        # prepare next input
        input_seq = torch.tensor([[char2idx[ch] for ch in generated[-seq_length:]]]).long()

print("\nGenerated text:")
print(generated)
