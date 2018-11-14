import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=42, fc_units=[32,32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state, defaults to the global state size from the env
            action_size (int): Dimension of each action, defaults to the global action size from the env
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in the hidden layers as a list
            ** Hard coded as a 3 layer network 
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.model = nn.Sequential(
            #nn.BatchNorm1d(state_size),
            nn.Linear(state_size,fc_units[0]),
            nn.ReLU(),
            #nn.BatchNorm1d(fc_units[0]),
            nn.Linear(fc_units[0],fc_units[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(fc_units[1]),
            nn.Linear(fc_units[1],action_size),
            nn.Tanh()
        )
        self.model.apply(self.init_weights)

    def init_weights(self,m):
        if (type(m) == nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(1.0)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.model(state)
    
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size , action_size , seed=42, fc_units=[32,32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state, defaults to the global state size from the env
            action_size (int): Dimension of each action, defaults to the global action size from the env
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in the hidden layers as a list
            ** Hard coded as a 3 layer network 
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hc_1 = nn.Sequential(
            nn.Linear(state_size,fc_units[0]),
            nn.ReLU(), # leaky relu ?
            nn.BatchNorm1d(fc_units[0])
        )
        self.hc_2 = nn.Sequential(
            nn.Linear(fc_units[0]+action_size,fc_units[1]),
            nn.ReLU(), # leaky relu ?
            nn.Linear(fc_units[1],1)
        )
        # Initialize the layers
        self.hc_1.apply(self.init_weights)
        self.hc_2.apply(self.init_weights)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.hc_1(state)
        x = torch.cat((xs, action), dim=1)
        x = self.hc_2(x)
        return (x)
    
    def init_weights(self,layer):
        if (type(layer) == nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(1.0)



