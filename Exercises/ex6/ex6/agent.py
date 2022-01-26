import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        
        self.value_layer=torch.nn.Linear(self.hidden, 1)

        # self.sigma = torch.zeros(1)  # TODO: Implement learned variance (or copy from Ex5)
        self.sigma = torch.nn.Parameter(torch.tensor([100.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, variance):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = variance  # TODO: Implement (or copy from Ex5)

        # Critic part
        # TODO: Implement
        state_value=self.value_layer(x)
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        action_dist=Normal(action_mean,torch.sqrt(sigma))
        # TODO: Return state value in addition to the distribution

        return action_dist,state_value


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.variance=self.policy.sigma
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []
        # self.values=[]

    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        

        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []
        # self.values=[]

        # TODO: Compute state values

        _, next_values = self.policy.forward(next_states,self.variance)
        _, allvalues=self.policy.forward(states,self.variance)

        # print("next_values",next_values.shape)
        # print("allvalues",allvalues.shape)
        # TODO: Compute critic loss (MSE)
        # Advantage estimates
        # TODO: Compute advantage estimates
        
        # td_target=rewards+self.gamma*next_values*(1-done)
        # c_loss=0.5*(td_target-allvalues).pow(2).mean()
        
        next_values=next_values.reshape(-1)
        allvalues=allvalues.reshape(-1)
        # print(rewards,self.gamma*next_values*(1-done))
        # print(allvalues)
        Q_pred=rewards+self.gamma*next_values*(1-done)
        # print(Q_pred)
        c_loss=torch.sum((Q_pred-allvalues).pow(2))
        advantages=Q_pred-allvalues

        # rewards=discount_rewards(rewards,self.gamma)
        # rewards=(rewards-torch.mean(rewards))/torch.std(rewards)
        # advantages=rewards-allvalues
        # c_loss=0.5*advantages.pow(2).mean()


        # TODO: Calculate actor loss (very similar to PG)
        a_loss=torch.sum(-(advantages.detach())*action_probs)
        # a_loss=(-(advantages.detach())*action_probs).mean()
        # print(c_loss,a_loss)
        
        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        ac_loss=a_loss+c_loss
        self.optimizer.zero_grad()
        ac_loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network
        # Or copy from Ex5
        a_dis , value =self.policy.forward(x,self.variance)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        # Or copy from Ex5
        if evaluation:
            action = a_dis.mean
        else:
            action = a_dis.sample((1,))[0] 
            # print(action[0])
        # TODO: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob=a_dis.log_prob(action)

        # store values for later use
        # self.values.append(value)

        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
