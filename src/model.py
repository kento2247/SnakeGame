from torch import nn
import torch.nn.functional as F
import torch


class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_size, 128).to(self.device)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.fc3 = nn.Linear(64, 32).to(self.device)
        self.fc4 = nn.Linear(32, action_size).to(self.device)
        self.dropout = nn.Dropout(0.2)

    def forward(self, state):
        state = state.to(self.device)
        x = self.fc1(state)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)
