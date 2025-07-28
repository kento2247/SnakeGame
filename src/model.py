from torch import nn
import torch.nn.functional as F
import torch


class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_size, 64).to(self.device)
        self.fc2 = nn.Linear(64, action_size).to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = self.fc1(state)
        x = F.relu(x)
        return self.fc2(x)
