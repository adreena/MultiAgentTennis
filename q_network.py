import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, seed, device):
        super(ActorNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        torch.manual_seed(seed)
        
        hidden_size_1 = 128
        hidden_size_2 = 64
        hidden_size_3 = 32
        hidden_size_4 = 16

        self.fc1 = torch.nn.Linear(state_size, hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size_2)
        self.fc3 = torch.nn.Linear(hidden_size_2, action_size)
        self.bn3 = torch.nn.BatchNorm1d(action_size)
        self.init_weights()
        
    def init_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        
        output = F.relu(self.fc1(state))
        # output = self.bn1(output)
        
        output = F.relu(self.fc2(output))
        # output = self.bn2(output)
        
        output = self.fc3(output)
        # output = self.bn3(output)
        return torch.tanh(output)
    
    
# class CriticNetwork(torch.nn.Module):
#     def __init__(self, state_size, action_size, seed,num_agents, device):
#         super(CriticNetwork, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         torch.manual_seed(seed)
        
#         hidden_size_1 = 128
#         hidden_size_2 = 64
#         hidden_size_3 = 32
#         # hidden_size_4 = 16

#         input_size = (state_size+action_size)*num_agents
#         self.bn0 = torch.nn.BatchNorm1d(input_size)
#         self.fc1 = torch.nn.Linear(input_size, hidden_size_1)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_size_1)
#         self.fc2 = torch.nn.Linear(hidden_size_1+ (action_size), hidden_size_2)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_size_2)
#         self.fc3 = torch.nn.Linear(hidden_size_2, 1)
#         # self.fc4 = torch.nn.Linear(hidden_size_3, hidden_size_4)
#         # self.fc5 = torch.nn.Linear(hidden_size_3, 1)
#         self.init_weights()
        
#     def init_weights(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
#         # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
#     def forward(self, states, actions):
#         output = torch.cat((states, actions), dim=1)
#         # output = self.bn0(output)
#         output = F.relu(self.fc1(output))
#         # output = self.bn1(output)
#         # output = torch.cat([output, action], 1)
#         output = F.relu(self.fc2(output))
#         # output = self.bn2(output)
#         # output = F.relu(self.fc3(output))
#         # output = F.relu(self.fc4(output))
#         return self.fc3(output)
                        
# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)





class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, num_agents, state_size, action_size, seed,device, fc1_units=128, fc2_units=68):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        input_size = (state_size+action_size)*num_agents
        # print('input_size', input_size)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (states, actions) pairs -> Q-values."""
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

