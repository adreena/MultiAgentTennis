import torch
import numpy as np
import random
from q_network import ActorNetwork,CriticNetwork 
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple
class DDPG:
    def __init__(self, num_agents, state_size, action_size, capacity, batch_size, learning_rate, update_rate, gamma, tau, device, seed, epsilon):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.actor = ActorNetwork(state_size, action_size, seed, device)
        self.critic = CriticNetwork(state_size, action_size, seed, device)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)
        
        self.memory = ReplayBuffer(capacity, seed, device)
        self.noise = OUNoise(action_size, seed)
        
        self.batch_size = batch_size
        self.device = device
        self.update_rate = update_rate
        
        self.step_ = 0.
        self.gamma = gamma
        self.tau = tau
        
    def reset(self):
        self.noise.reset()
        
    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(dim=0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state) #.cpu().data.numpy() #.squeeze(dim=0)
        self.actor.train()
        
        noise = torch.from_numpy(self.noise.sample()).float()
        action += noise #* epsilon
        action = torch.clamp(action,-1.0,1.0)
        return action


    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)
        
        self.step_=(self.step_+1)%self.update_rate
        if self.step_ == 0 and len(self.memory) > self.batch_size:
            self.learn()
        
            
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
       
        q_expected_val = self.critic_target(states, actions)
        
        next_actions = self.actor_target(next_states)
        q_target_next = self.critic_target(next_states,next_actions)
        q_target_val = rewards + self.gamma*q_target_next*(1-dones)

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(q_expected_val, q_target_val)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        action_pred = self.actor(states)
        actor_loss = - self.critic(states,action_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
    def soft_update(self, target, local):
        for target_param , local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
