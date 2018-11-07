import torch
import numpy as np
import random
from q_network import ActorNetwork,CriticNetwork 
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy

class MADDPG:
    def __init__(self, num_agents, state_size, action_size, capacity, batch_size, learning_rate, update_rate, gamma, tau, device, seed, epsilon):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.actors = [ActorNetwork(state_size, action_size, seed, device) for agent in range(num_agents)]
        self.critics = [CriticNetwork(state_size, action_size, seed, device) for agent in range(num_agents)]
        self.critics_target = deepcopy(self.critics)
        self.actors_target = deepcopy(self.actors)
        
        self.actors_optimizer = [torch.optim.Adam(actor.parameters(), lr = learning_rate) for actor in self.actors]
        self.critics_optimizer = [torch.optim.Adam(critic.parameters(), lr = learning_rate) for critic in self.critics]
        
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
        
    def act(self, states):
        # state = torch.from_numpy(state)
        # self.actor.eval()
        # with torch.no_grad():
        #     action = self.actor(state).cpu().data.numpy()
        # self.actor.train()
        # action += self.noise.sample()
        # return action
        actions = torch.zeros(self.num_agents, self.action_size)
        for agent_idx in range(self.num_agents):
            st = torch.from_numpy(states[agent_idx,:]).float().unsqueeze(dim=0)
            self.actors[agent_idx].eval()
            with torch.no_grad():
                action = self.actors[agent_idx](st).squeeze(dim=0)
            self.actors[agent_idx].train()
            # action += self.noise.sample()
            actions[agent_idx,:] = action
        return actions

        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.step_=(self.step_+1)%self.update_rate
        if self.step_==0 and len(self.memory)> self.batch_size:
            self.learn()
            
    def learn(self):
        for agent_idx in range(self.num_agents):
            samples = self.memory.sample(self.batch_size, self.device)
            states, actions, rewards, next_states, dones = samples

            next_actions = self.actors_target[agent_idx](next_states)
            q_target_next = self.critics_target[agent_idx](next_states, next_actions)
            q_target_val = rewards + self.gamma*q_target_next*(1-dones)
            q_expected_val = self.critics[agent_idx](states, actions)

            critic_loss = F.mse_loss(q_expected_val, q_targte_val)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_preds = self.actors[agent_idx](states)
            actor_loss = - self.critics[agent_idx](state,actor_preds).mean()
            self.actors_optimizer[agent_idx].zero_grad()
            actor_loss.backward()
            self.actors_optimizer[agent_idx].step()
            
            self.soft_update(self.actors_target[agent_idx], self.actors[agent_idx])
            self.soft_update(self.critics_target[agent_idx], self.critics[agent_idx])
        
    def soft_update(self, target, local):
        for target_param , local_param in zip(target.parameters(), local.parameters()):
            targey_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
