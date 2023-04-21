import torch
import torch.nn.functional as F
import math

class ReconstructionLoss(torch.nn.Module):
   def __init__(self):
      super(ReconstructionLoss, self).__init__()
   def forward(self, fake, real):
      fake = fake[:,:,:-1]
      real = real[:,:,:-1]
      # This should probably be a threshold, i.e. if its too close penalise, otherwise don't.
      # Doing it this way encourages it to be as different as possible, instead of just not being too close.
      # Possible solution is to scale it e.g. 1 for very close, but tends to 0 very quickly (e.g. difference of 0.25 is 0.0000001) or w/e
      sim_matrix = torch.squeeze(F.cosine_similarity(fake, real), dim=-1)
      # Resulting graph (f and g respectively) https://www.desmos.com/calculator/tiuvnazff6
      loss_f = torch.sum(sim_matrix)
      alpha = 0.2
      beta = 3 / (math.log((alpha+1)/alpha))
      loss_g = alpha*((math.e)**(loss_f / beta) - 1)
      return loss_g
   
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