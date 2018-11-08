import torch
import numpy as np
import random
from q_network import ActorNetwork,CriticNetwork 
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple
class MADDPG:
    def __init__(self, num_agents, state_size, action_size, capacity, batch_size, learning_rate, update_rate, gamma, tau, device, seed, epsilon):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.actors = [ActorNetwork(state_size, action_size, seed, device) for agent in range(num_agents)]
        self.critics = [CriticNetwork(state_size, action_size, num_agents, seed, device) for agent in range(num_agents)]
        self.critics_target = deepcopy(self.critics)
        self.actors_target = deepcopy(self.actors)
        
        self.actors_optimizer = [torch.optim.Adam(actor.parameters(), lr = learning_rate) for actor in self.actors]
        self.critics_optimizer = [torch.optim.Adam(critic.parameters(), lr = learning_rate) for critic in self.critics]
        
        self.memory = ReplayBuffer(capacity, seed, device)
        self.noise = [OUNoise(action_size, seed) for agent in range(num_agents)]
        
        self.batch_size = batch_size
        self.device = device
        self.update_rate = update_rate
        
        self.step_ = 0.
        self.gamma = gamma
        self.tau = tau
        self.var = [1.0 for i in range(num_agents)]
        self.eps_done = 0
        self.eps_b_train = 3000
        
    def reset(self):
        for i in range(self.num_agents):
            self.noise[i].reset()
        
    def act(self, states, epsilon):

        actions = torch.zeros(self.num_agents, self.action_size)
        for agent_idx in range(self.num_agents):
            st = torch.from_numpy(states[agent_idx,:]).float().unsqueeze(dim=0)
            self.actors[agent_idx].eval()
            with torch.no_grad():
                action = self.actors[agent_idx](st).squeeze(dim=0)
            self.actors[agent_idx].train()
#             noise_val = torch.from_numpy(np.array(self.noise[agent_idx].sample())).float()
            action = self.add_noise2(action, agent_idx)
#             action += noise_val * epsilon
            action= torch.clamp(action,-1.0,1.0)
            actions[agent_idx,:] = action
        return actions

    def add_noise2(self, action, i):
#         FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor
        action += torch.from_numpy(np.random.randn(2) * self.var[i]).float()

        if self.eps_done > self.eps_b_train and self.var[i] > 0.05:
            self.var[i] *= 0.999998
        #action = th.clamp(action, -1.0, 1.0)

        return action   
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.step_=(self.step_+1)%self.update_rate
        if self.step_ == 0 and len(self.memory) > self.batch_size:
            self.learn()
        
            
    def learn(self):
        #st: torch.Size([1, 24])
        #action: torch.Size([2, 24])

        for agent_idx in range(self.num_agents):
            samples = self.memory.sample(self.batch_size, self.device)
            
            states, actions, rewards, next_states, dones = samples
         
            full_states = states.view(self.batch_size,-1)
            full_actions = actions.view(self.batch_size,-1)
            full_next_states = next_states.view(self.batch_size,-1)
           
            q_expected_val = self.critics_target[agent_idx](full_states, full_actions)
            
            
            next_actions = [self.actors_target[agent_idx](next_states[:,i,:]) for i in range(self.num_agents)]
            next_actions = torch.stack(next_actions)
            next_actions =  next_actions.view(self.batch_size,-1)
            next_actions = (next_actions.transpose(0,1).contiguous())

            next_actions= next_actions.view(-1,self.num_agents * self.action_size)
            next_states = next_states.view(-1, self.num_agents * self.state_size)
            q_target_next = self.critics_target[agent_idx](next_states,next_actions)

            q_target_val = rewards[:,agent_idx]*0.01 + self.gamma*q_target_next*(1-dones[:,agent_idx])
            

            self.critics_optimizer[agent_idx].zero_grad()
            critic_loss = F.mse_loss(q_expected_val, q_target_val.detach())
            critic_loss.backward()
            self.critics_optimizer[agent_idx].step()
            
            self.actors_optimizer[agent_idx].zero_grad()
            action_pred = self.actors[agent_idx](states[:,agent_idx,:])
            action_clone = actions.clone()
            action_clone[:,agent_idx,:] = action_pred
            actor_loss = - self.critics[agent_idx](full_states,action_clone.view(self.batch_size, -1)).mean()
            actor_loss.backward()
            self.actors_optimizer[agent_idx].step()

            self.soft_update(self.actors_target[agent_idx], self.actors[agent_idx])
            self.soft_update(self.critics_target[agent_idx], self.critics[agent_idx])
        
    def soft_update(self, target, local):
        for target_param , local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
