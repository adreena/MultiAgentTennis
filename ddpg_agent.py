import torch
import numpy as np
import random
from q_network import ActorNetwork,CriticNetwork 
from replay_buffer import ReplayBuffer
from noise import OUNoise
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple

class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""

    def __init__(self, num_agents, state_size, action_size, capacity,
                   batch_size,actor_learning_rate, critic_learning_rate, update_rate, gamma, tau, 
                   device, seed, epsilon, epsilon_min, epsilon_decay):
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.device=device
        self.buffer_size = capacity
        self.batch_size = batch_size
        self.update_every = update_rate
        self.gamma = gamma
        self.num_agents = num_agents
        # self.noise_weight = noise_start
        # self.noise_decay = noise_decay
        self.t_step = 0

        # create two agents, each with their own actor and critic
        self.agents = []
        for i in range(num_agents):
            self.agents.append(DDPG(id = i, state_size=state_size, action_size=action_size,\
                                     actor_learning_rate=actor_learning_rate,\
                                     critic_learning_rate=critic_learning_rate,\
                                     num_agents=num_agents,gamma=gamma, tau=tau, device=device, seed=seed,\
                                     epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay))
        
        # shared replay buffer
        self.memory = ReplayBuffer(self.buffer_size, seed, device)
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0  and len(self.memory) > self.batch_size:
            experiences = [self.memory.sample(batch_size=self.batch_size, device=self.device) for _ in range(self.num_agents)]
            self.learn(experiences, self.gamma)

    def act(self, states):
        all_actions = []
        for agent_idx in range(self.num_agents):
            state = states[agent_idx]
            agent = self.agents[agent_idx]
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(dim=0)
            action = agent.act(state) * self.epsilon
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    # def learn(self, experiences, gamma):
    #     # each agent uses it's own actor to calculate next_actions
    #     next_actions = []
    #     full_actions = []
    #     states = []
    #     next_states = []
    #     rewards = []
    #     actions = []
    #     dones = []
    #     for agent_idx in range(len(self.agents)):
    #         _states, _actions, _rewards, _next_states, _dones = experiences[agent_idx]


    #         temp_next_states = _next_states.reshape(-1, 2, 24)[:,agent_idx].squeeze(1)
    #         _next_actions = self.agents[agent_idx].actor_target(temp_next_states)
    #         temp_states = _states.reshape(-1, 2, 24)[:,agent_idx].squeeze(1)
    #         action_pred = self.agents[agent_idx].actor(temp_states)

    #         next_actions.append(_next_actions)
    #         full_actions.append(action_pred)
    #         states.append(_states)
    #         dones.append(_dones[:,agent_idx].unsqueeze(1))
    #         rewards.append(_rewards[:,agent_idx].unsqueeze(1))
    #         actions.append(_actions)
    #         next_states.append(_next_states)

            
    #     # # each agent uses it's own actor to calculate actions
    #     # all_actions = []
    #     # for i, agent in enumerate(self.agents):
    #     #     states, _, _, _, _ = experiences[i]
    #     #     agent_id = torch.tensor([i]).to(device)
            
    #     #     action = agent.actor_local(state)
    #     #     all_actions.append(action)
    #     # each agent learns from it's experience sample
    #     next_actions = torch.cat(next_actions, dim=1).to(self.device)
    #     for agent_idx in range(len(self.agents)):
    #         self.agents[agent_idx].learn(agent_idx, states[agent_idx], actions[agent_idx], \
    #             rewards[agent_idx], next_states[agent_idx], dones[agent_idx], next_actions, full_actions)

    #     self.epsilon -= self.epsilon_decay
    #     self.epsilon = max(self.epsilon_min, self.epsilon)


    def learn(self, experiences, gamma):
        # each agent uses it's own actor to calculate next_actions
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(self.device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
        # each agent uses it's own actor to calculate actions
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(self.device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor(state)
            all_actions.append(action)
        # each agent learns from it's experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)

class DDPG:
    def __init__(self,id, state_size, action_size, actor_learning_rate, num_agents,
        critic_learning_rate, gamma, tau, device, seed, epsilon, epsilon_min, epsilon_decay):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.actor = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device)
        self.critic = CriticNetwork(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=seed, device=device)
        self.critic_target = CriticNetwork(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=seed, device=device)
        self.actor_target = ActorNetwork(state_size=state_size, action_size=action_size, seed=seed, device=device)
        
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_learning_rate)
        self.tau = tau
        self.gamma = gamma
        self.noise = OUNoise(action_size, seed)
        self.device = device

    def reset(self):
        self.noise.reset()
        
    def act(self, state):
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy() #.squeeze(dim=0)
        self.actor.train()
        
        noise = self.noise.sample()
        action += noise
        action = np.clip(action,-1.0,1.0)
        return action
        
    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by it's actor)
            all_actions (list): each agent's action (as calculated by it's actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(self.device)
        actions_next = torch.cat(all_next_actions, dim=1).to(self.device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic(states, actions)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        self.critic_loss = critic_loss.item()  # for tensorboard logging
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(self.device)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_loss = actor_loss.item()  # calculate policy gradient
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update( self.critic_target,self.critic)
        self.soft_update( self.actor_target,self.actor)
            
    # def learn(self,agent_idx, states, actions, rewards, next_states, dones, next_actions, full_actions):
    #     # print('STATES:',states.size())
    #     # print('ACTIONS:', actions.size())
    #     # print('REWARDS:',rewards.size())
    #     # print('NEXT_STATES:',next_states.size())
    #     # print('DONES:',dones.size())
    #     # print('NEXT_ACTIONS:', next_actions.size())

        
    #     # next_states = next_states.reshape(-1, 2, 24)[:,agent_idx].squeeze(1)
    #     with torch.no_grad():
    #         q_target_next = self.critic_target(next_states, next_actions)
    #     q_target_val = rewards + self.gamma*q_target_next*(1-dones)
    #     q_expected_val = self.critic(states, actions)


    #     self.critic_optimizer.zero_grad()
    #     critic_loss = F.mse_loss(q_expected_val, q_target_val.detach())
    #     # print('critic_loss', critic_loss)
    #     critic_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
    #     self.critic_optimizer.step()
        


    #     #---------------------------------------
    #     self.actor_optimizer.zero_grad()
    #     action_pred = [actions if i == agent_idx else actions.detach() for i, actions in enumerate(full_actions)]
    #     action_pred = torch.cat(action_pred, dim=1).to(self.device)

    #     actor_loss = - self.critic(states,action_pred).mean()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()

    #     self.soft_update(self.actor_target, self.actor)
    #     self.soft_update(self.critic_target, self.critic)

        
    #     # self.noise.reset()

        
    def soft_update(self, target, local):
        for target_param , local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
   
            
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
