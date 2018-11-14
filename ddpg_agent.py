import torch
import numpy as np
import random
from q_network import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple

# Number of neurons in the layers of the Actor & Critic Networks
FC_UNITS_ACTOR = [32,32] # [16,16] # [8,8] # [64,64] # [32,16] # [128,128] # [64,128] # [32,16] # [400,300] # [128,128]
FC_UNITS_CRITIC = [32,32] 
class DDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, buffer_size, batch_size, actor_learrning_rate, critic_learning_rate, gamma, tau, optimizer_weight_decay, device, seed=42):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(state_size, action_size, seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learrning_rate)

        self.critic = Critic(state_size, action_size, seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, seed).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=optimizer_weight_decay)

        self.noise = OUNoise(action_size, seed)
        self.memory = ReplayBuffer(buffer_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, add_noise=False):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # critic ---------------------------- #
        next_actions = self.actor_target(next_states)
        q_target_next = self.critic_target(next_states, next_actions)
        q_target = rewards + (self.gamma * q_target_next * (1 - dones))
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor ---------------------------- #
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks ----------------------- #
        self.soft_update(local_model=self.critic, target_model=self.critic_target)
        self.soft_update(local_model=self.actor, target_model=self.actor_target)                     

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
      