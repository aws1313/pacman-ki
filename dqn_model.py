import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, inp_size: int, hidden_size, out_size: int):
        super(DQN, self).__init__()
        self.rls = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.rls(x)

    def save(self):
        torch.save(self.state_dict(), './model.pth')
