import torch
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
   def __init__(self):
      super(ReconstructionLoss, self).__init__()
   def forward(self, fake, real):
      sim_matrix = torch.squeeze(F.cosine_similarity(fake, real), dim=-1)
      loss = torch.sum(sim_matrix)
      return loss
   
class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Generator, self).__init__()
        self.model = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.float()

    def forward(self, x):
      x = x.to(torch.float32)
      out, (h, c) = self.model(x)
      out = self.fc(out)
      # add pitch
      pitch = x[:,:,-1:]
      out = torch.cat([out, pitch], dim=-1)
      return out
    
class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.model = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.float()

    def forward(self, x):
      x = x.to(torch.float32)
      out, (h, c) = self.model(x)
      out = self.fc(out)
      return out