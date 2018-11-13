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
    def __init__(self, state_size, action_size, actor_learning_rate, update_rate,
        critic_learning_rate, gamma, tau, device, seed, epsilon, epsilon_min, epsilon_decay, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
#         self.seed = random.seed(seed)
        
        weight_decay = 1e-4
        self.actor = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
        self.actor_target = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
#         self.hard_update(self.actor_target, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_learning_rate)

        self.critic = CriticNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
        self.critic_target = CriticNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
#         self.hard_update(self.critic_target, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_learning_rate, weight_decay=weight_decay)

        self.tau = tau
        self.gamma = gamma
        self.noise = OUNoise(action_size, seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(capacity=buffer_size, seed=self.seed, device=self.device)
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay 
        

    def reset(self):
        self.noise.reset()
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy() #.squeeze(dim=0)
        self.actor.train()
        
        noise = self.noise.sample()
        action += noise #* self.epsilon
        action = np.clip(action,-1,1)
        return action
    
    
    def step(self,state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(batch_size=self.batch_size, device=self.device)
            self.learn(experiences)
        
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by it's actor)
            all_actions (list): each agent's action (as calculated by it's actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # critic ---------------------------- #
        
        next_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor ---------------------------- #
        
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update( self.critic_target,self.critic)
        self.soft_update( self.actor_target,self.actor)
        
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        
        
    def soft_update(self, target, local):
        for target_param , local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
