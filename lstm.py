import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.model = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.double()

    def forward(self, x):
      out, (h, c) = self.model(x)
      out = self.fc(out)
      # add pitch
      pitch = x[:,:,-1:]
      out = torch.cat([out, pitch], dim=-1)
      print(out.shape)
      return out

