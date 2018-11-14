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
#         self.fc1 = nn.Linear(state_size,32)
#         self.fc2 = nn.Linear(32, 32),
#         self.fc3 = nn.Linear(32,action_size)
        
        self.fc1 = torch.nn.Linear(state_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, action_size)
        
#         self.model.apply(self.init_weights)

    def init_weights(self,m):
        if (type(m) == nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(1.0)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        return torch.tanh(self.fc3(output))
        
    
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size , action_size , seed=42, fc_units=[32,32]):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        
        self.fc1 = torch.nn.Linear(state_size, 32)
        self.fc2 = torch.nn.Linear(32+action_size, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        
#         self.hc_1.apply(self.init_weights)
#         self.hc_2.apply(self.init_weights)

    def forward(self, state, action):
        output = F.relu(self.fc1(state))
        output = torch.cat((output, action), dim=1)
        output = F.relu(self.fc2(output))
        return self.fc3(output)
    
    def init_weights(self,layer):
        if (type(layer) == nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(1.0)



