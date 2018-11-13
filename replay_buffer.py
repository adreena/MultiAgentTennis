from collections import deque, namedtuple
import numpy as np
import random
import torch 
class ReplayBuffer():
    def __init__(self, capacity, seed, device):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=("state", "action", "reward", "next_state", "done"))
        self.seed = seed
        self.device = device
        
    def sample(self, batch_size, device):
        
        experiences = random.sample(self.buffer, k = batch_size)
        states=torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)

        next_states=torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        rewards=torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones=torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        actions=torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    def transpose_list(mylist):
        return list(map(list, zip(*mylist)))
    
    def add(self, state, action, reward, next_state, done):
#         trainsition = 
        new_experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(new_experience)
        
    def __len__(self):
        return len(self.buffer)
