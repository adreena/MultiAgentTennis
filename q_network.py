import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_size_1 = 32
        hidden_size_2 = 32
#         hidden_size_3 = 32
#         hidden_size_4 = 16
        self.fc1 = torch.nn.Linear(state_size, hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
#         self.fc3 = torch.nn.Linear(hidden_size_2, hidden_size_3)
#         self.fc4 = torch.nn.Linear(hidden_size_3, hidden_size_4)
        self.fc5 = torch.nn.Linear(hidden_size_2, action_size)


    def forward(self, state):
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
#         output = F.relu(self.fc3(output))
#         output = F.relu(self.fc4(output))
        return torch.tanh(self.fc5(output))
        
    
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size , action_size , seed=42):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_size_1 = 32
        hidden_size_2 = 32
        
        self.fc1 = torch.nn.Linear(state_size, hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1+action_size, hidden_size_2)
        self.fc3 = torch.nn.Linear(hidden_size_2, 1)
        
    def forward(self, state, action):
        output = F.relu(self.fc1(state))
        output = torch.cat((output, action), dim=1)
        output = F.relu(self.fc2(output))
        return self.fc3(output)
    
    def init_weights(self,layer):
        if (type(layer) == nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(1.0)



