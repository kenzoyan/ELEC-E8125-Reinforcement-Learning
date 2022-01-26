import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from utils import discount_rewards  # from utils.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        
        # self.sigma = torch.nn.Parameter(torch.tensor([25.]))  # TODO: Implement accordingly T1
        self.sigma = torch.nn.Parameter(torch.tensor([100.0]))  # TODO: Implement accordingly (T2)  

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, variance):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = variance  # TODO: Is it a good idea to leave it like this? ??

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        action_dist=Normal(action_mean,torch.sqrt(sigma))
        return action_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.baseline=20   #  IN task 1 b
        self.variance=self.policy.sigma

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # compute decaying variance 
        # c=5e-4
        # self.variance= self.policy.sigma*np.exp(-c* episode_number )
        # print("self.variance",self.variance)
        # TODO: Compute discounted rewards
        rewards=discount_rewards(rewards,self.gamma)
        
        rewards=(rewards-torch.mean(rewards))/torch.std(rewards)  # normalization 

        # TODO: Compute the optimization term (T1)
        loss=torch.sum(-(rewards)*action_probs)  #without baseline

        # loss=torch.sum(-(rewards-self.baseline)*action_probs)  # with baseline

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)


        # TODO: Pass state x through the policy network (T1)
        
        a_dis=self.policy.forward(x,self.variance)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        
        if evaluation:
            action = a_dis.mean
        else:
            action = a_dis.sample((1,))[0] 
            # print(action[0])

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob=a_dis.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

