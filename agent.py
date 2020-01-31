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
        # self.sigma = 5  # DONE: constant variance, T1
        # self.sigma_init = torch.tensor([10.])  # DONE: initial value, T2a
        # self.sigma = torch.tensor([10.]) # T2a
        self.sigma = torch.nn.Parameter(torch.tensor([10.])) # T2b
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
        # sigma = self.sigma # T1
        # sigma = F.softplus(self.sigma_init) # T2a
        sigma = F.softplus(self.sigma) # T2

        # DONE: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        dist = Normal(mu, sigma)
        return dist

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []
        # b = 20 # T1b baseline

        # T2a
        # self.policy.sigma = self.policy.sigma_init*np.exp(-0.0005*episode_number)

        # DONE: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards) # T1c
        discounted_rewards /= torch.std(discounted_rewards) # T1c

        weighted_probs = -action_probs * discounted_rewards # T1a, T1c, T2
        # weighted_probs = -action_probs * (discounted_rewards - b) # T1b

        # DONE: Compute the optimization term (T1)
        loss = torch.mean(weighted_probs)

        # DONE: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # DONE: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # DONE: Pass state x through the policy network (T1)
        dist = self.policy.forward(x)

        # DONE: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action =  torch.mean(dist)
        else:
            action = dist.sample()

        # DONE: Calculate the log probability of the action (T1)
        act_log_prob = dist.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

