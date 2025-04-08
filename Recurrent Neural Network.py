import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
text = "hellohellohello"
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
input_size = len(chars)
hidden_size = 8
output_size = len(chars)
seq_length = 5
lr = 0.01
num_epochs = 100

# Prepare dataset
x_data = []
y_data = []

for i in range(len(text) - seq_length):
    x_str = text[i:i+seq_length]
    y_str = text[i+1:i+seq_length+1]
    x_data.append([char2idx[c] for c in x_str])
    y_data.append([char2idx[c] for c in y_str])

x_tensor = torch.Tensor(x_data).long()
y_tensor = torch.Tensor(y_data).long()

# RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # One-hot encoding
        x = torch.nn.functional.one_hot(x, num_classes=input_size).float()
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs.view(-1, output_size), y_tensor.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Inference
with torch.no_grad():
    start = "hell"
    input_seq = torch.tensor([[char2idx[ch] for ch in start]]).long()
    for _ in range(10):
        output = model(input_seq)
        last_char_logits = output[0, -1]
        predicted = torch.argmax(last_char_logits).item()
        start += idx2char[predicted]
        input_seq = torch.tensor([[char2idx[ch] for ch in start[-seq_length:]]])

    print("Predicted sequence:", start)
