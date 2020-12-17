import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Neural netowrk to predict Q values"""

    def __init__(self, 
                 state_size, 
                 action_size, 
                 seed, 
                 hidden_dim_1=64, 
                 hidden_dim_2=128,
                 ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_dim_1: number of nodes in the first hidden layer
            hidden_dim_2: number of nodes in the second layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x