import torch
import numpy as np
import random
from q_network import Actor,Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of neurons in the layers of the Actor & Critic Networks
FC_UNITS_ACTOR = [32,32] # [16,16] # [8,8] # [64,64] # [32,16] # [128,128] # [64,128] # [32,16] # [400,300] # [128,128]
FC_UNITS_CRITIC = [32,32] 
class DDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, buffer_size, batch_size, actor_learrning_rate, critic_learning_rate, gamma, tau, optimizer_weight_decay, seed=42):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learrning_rate)

        self.critic = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=optimizer_weight_decay)

        self.noise = OUNoise(action_size, seed)
        self.memory = ReplayBuffer(buffer_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(local_model=self.critic, target_model=self.critic_target)
        self.soft_update(local_model=self.actor, target_model=self.actor_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
# class DDPG:
#     def __init__(self, state_size, action_size, actor_learning_rate, update_rate,
#         critic_learning_rate, gamma, tau, device, seed, epsilon, epsilon_min, epsilon_decay, buffer_size, batch_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(seed)
        
#         weight_decay = 1e-4
#         self.actor = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
#         self.actor_target = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
# #         self.hard_update(self.actor_target, self.actor)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_learning_rate)

#         self.critic = CriticNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
#         self.critic_target = CriticNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device).to(device)
# #         self.hard_update(self.critic_target, self.critic)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_learning_rate, weight_decay=weight_decay)

#         self.tau = tau
#         self.gamma = gamma
#         self.noise = OUNoise(action_size, seed)
#         self.device = device
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.memory = ReplayBuffer(capacity=buffer_size, seed=seed, device=self.device)
#         self.epsilon = epsilon 
#         self.epsilon_min = epsilon_min 
#         self.epsilon_decay = epsilon_decay 
        

#     def reset(self):
#         self.noise.reset()
        
#     def act(self, state):
#         state = torch.from_numpy(state).float().to(self.device)
#         self.actor.eval()
#         with torch.no_grad():
#             action = self.actor(state).cpu().data.numpy()
#         self.actor.train()
# #         if add_noise:
# #             action += self.noise.sample()
#         return np.clip(action, -1, 1)
    
    
#     def step(self,state, action, reward, next_state, done):
#         self.memory.add(state, action, reward, next_state, done)

#         # Learn, if enough samples are available in memory
#         if len(self.memory) > self.batch_size:
#             experiences = self.memory.sample(batch_size=self.batch_size, device=self.device)
#             self.learn(experiences)
        
#     def learn(self, experiences):
#         """Update policy and value parameters using given batch of experience tuples.

#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#             gamma (float): discount factor
#             all_next_actions (list): each agent's next_action (as calculated by it's actor)
#             all_actions (list): each agent's action (as calculated by it's actor)
#         """

#         states, actions, rewards, next_states, dones = experiences

#         # critic ---------------------------- #
        
#         next_actions = self.actor_target(next_states)
#         q_targets_next = self.critic_target(next_states, next_actions)
#         q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
#         q_expected = self.critic(states, actions)
#         critic_loss = F.mse_loss(q_expected, q_targets.detach())
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # actor ---------------------------- #
        
#         actions_pred = self.actor(states)
#         actor_loss = -self.critic(states, actions_pred).mean()
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update( self.critic_target,self.critic)
#         self.soft_update( self.actor_target,self.actor)
        
#         self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        
        
#     def soft_update(self, target, local):
#         for target_param , local_param in zip(target.parameters(), local.parameters()):
#             target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
#     def hard_update(self, target, source):
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(param.data)
