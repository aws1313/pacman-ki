import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, inp_size: int, hidden_size, out_size: int, lr, gamma, device):
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
        self.run_device = device

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.flatten(x)
        return self.rls(x)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def train_step(self, old_state, new_state, action, reward):
        # Da wir uns nicht sicher waren, ob unsere ursprüngliche Implementierung funktioniert hat, haben wir uns
        # an diesem Code von dem Spiel Snake orientiert und ihn für uns angepasst:
        # https://github.com/patrickloeber/snake-ai-pytorch/blob/main/model.py
        old_state = torch.from_numpy(old_state).to(self.run_device)
        new_state = torch.from_numpy(new_state).to(self.run_device)
        action = torch.tensor(action, dtype=torch.float).to(self.run_device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.run_device)
        if len(old_state.shape) == 1:
            old_state = torch.unsqueeze(old_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction = self(old_state)
        target_prediction = prediction.clone()

        for i in range(len(old_state)):
            Q_new = reward[i]
            Q_new = reward[i] + self.gamma * torch.max(self(new_state[i]))
            target_prediction[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(prediction, target_prediction)
        loss.backward()
        self.optimizer.step()
