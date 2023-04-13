import torch

class Transformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
      super(Transformer, self).__init__()
      self.hidden_size = hidden_size
      self.input_size = input_size
      self.num_layers = num_layers
      self.output_size = output_size

      self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
      self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
      x = x.to(torch.float32)
      out, hidden = self.rnn(x)
      out = self.fc(out)
      return out