import math

import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)
        # self.sigma = torch.tensor([math.sqrt(5)]) # T1
        # self.sigma_init = torch.tensor([10.0]) # T2a
        # self.sigma = torch.tensor([10.0]) # T2a
        self.sigma = torch.nn.Parameter(torch.tensor([10.0])) # T2b
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = self.sigma  # softplus can map from o to inf, not losing infor about correposn differences in values
        # Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        normal_dist = Normal(mu, sigma)

        # Add a layer for state value calculation (T3)
        value = self.fc2_value(x)

        return normal_dist, value

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.state_values = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.state_values = [], [], [], []

        # Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / torch.std(discounted_rewards)  # T1c

        # Compute critic loss and advantages (T3)
        loss = 0
        for log_prob, value, reward in zip(action_probs, state_values, discounted_rewards):
            advantage = reward - value.item()
            policy_loss = -advantage * log_prob
            value_loss = F.smooth_l1_loss(value, reward) # Using loss l1
            loss += (policy_loss + value_loss)

        # Compute the optimization term (T1, T3)
        #loss = policy_losses.sum() + value_losses.sum()
        #  Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        #  Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # Pass state x through the policy network (T1)
        policy_mean_distribution, state_value = self.policy.forward(x)

        #  Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = policy_mean_distribution.mean
        else:
            action = policy_mean_distribution.sample()

        #  Calculate the log probability of the action (T1)
        act_log_prob = policy_mean_distribution.log_prob(action)

        #  Return state value prediction, and/or save it somewhere (T3)
        return action, act_log_prob, state_value

    def store_outcome(self, observation, action_prob, action_taken, reward, state_value):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.state_values.append(state_value)
        self.rewards.append(torch.Tensor([reward]))

