import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, inp_size: int, hidden_size, out_size: int, lr, gamma):
        super(DQN, self).__init__()
        self.rls = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.gamma = gamma
        self.flatten = nn.Flatten()


    def forward(self, x):
        x=x.unsqueeze(0)
        x = self.flatten(x)
        return self.rls(x)

    def save(self):
        torch.save(self.state_dict(), './model.pth')

    def train_step(self, old_state, new_state, action, reward):
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        print(old_state.shape)
        if len(old_state.shape) == 1:
            old_state = torch.unsqueeze(old_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction = self(old_state)
        target_prediction = prediction.clone()

        for i in range(len(old_state)):
            Q_new = reward[i]
            Q_new = reward[i] + self.gamma * torch.max(self(new_state[i].unsqueeze(0)))
            target_prediction[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(prediction, target_prediction)
        loss.backward()
        self.optimizer.step()
