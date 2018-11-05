import torch
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Nerwork(torch.nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        torch.set_manual_seed(seed)
        
        hidden_size = 120
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_size)
        self.init_weights()
        
    def init_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        outout = F.relu(self.fc1(state))
        output = self.fc2(output)
        return output
                        