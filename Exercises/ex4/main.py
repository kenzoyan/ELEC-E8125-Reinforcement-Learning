import gym
import numpy as np
from matplotlib import pyplot as plt
from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
from dqn_agent import Agent as DQNAgent  # Task 4
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import plot_rewards
import seaborn as sb

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# env_name = "CartPole-v0"
env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Set hyperparameters
# Values for RBF (Tasks 1-3)
# glie_a = 50
# # num_episodes = 1000
# num_episodes = 1000

# Values for DQN  (Task 4)
if "CartPole" in env_name:
    TARGET_UPDATE = 50
    glie_a = 500
    num_episodes = 2000
    hidden = 12
    gamma = 0.95
    replay_buffer_size = 500000
    batch_size = 256
elif "LunarLander" in env_name:
    TARGET_UPDATE = 20
    glie_a = 5000
    num_episodes = 15000
    hidden = 64
    gamma = 0.95
    replay_buffer_size = 50000
    batch_size = 128
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)

# The output will be written to your folder ./runs/CURRENT_DATETIME_HOSTNAME,
# Where # is the consecutive number the script was run
writer = SummaryWriter("runs/batch32_hid12_gamma98_Adam_Hubber_glie250")

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Tasks 1-3 - RBF
# agent = RBFAgent(n_actions)

# Task 4 - DQN
agent = DQNAgent(env_name, state_space_dim, n_actions, replay_buffer_size, batch_size,
              hidden, gamma)

# Training loop
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    # Task 1: TODO: Implement GLIE for epsilon greedy
    # Hint: See Exercise 3
    eps = round(glie_a/(glie_a+ep),4)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values
        # print("______")
        # print(state,action,next_state,reward,done)
        # agent.single_update(state,action,next_state,reward,done)

        # Task 2: TODO: Store transition and batch-update Q-values
        # Hint: Use the single_update() function in Task 1 and the update_estimator() function in in Task 2.
        agent.store_transition(state,action,next_state,reward,done)     #s, a, s0, r,done
        # agent.update_estimator()  # implentment

        # Task 4: Update the DQN
        agent.update_network()
        # Move to the next state
        state = next_state
    if ep % 10 == 0:
        print("Episode {} ended, eps={} , cum_reward={}".format(ep,eps,cum_reward))
    cumulative_rewards.append(cum_reward)
    writer.add_scalar('Training ' + env_name, cum_reward, ep)
    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        torch.save(agent.policy_net.state_dict(),
                  "weights_%s_%d.mdl" % (env_name, ep))
        plot_rewards(cumulative_rewards)
        print('Saved a figure')
        plt.savefig("re_"+str(ep)+".png")
        plt.close()
        np.save("Reward_"+str(ep)+".npy",cumulative_rewards)

plot_rewards(cumulative_rewards)
print('Complete')
plt.ioff()
plt.show()

# Task 3 - plot the policy

# Code from Ex 3

# Reasonable values for Cartpole discretization
discr = 32
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr))

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av

def get_cell_index2(x,th):
    x = find_nearest(x_grid, x)
    # v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, th)
    # av = find_nearest(av_grid, state[3])
    return x, th,

x_grid=x_grid.round(2)
th_grid=th_grid.round(2)

for i,x in enumerate(x_grid):
    for j,th in enumerate(th_grid):
        
        x,th=get_cell_index2(x,th)
        state=np.array([x,0,th,0])
        action=agent.get_action(state)
        q_grid[(i,j)]=action
    
# print(q_grid,x_grid,th_grid)

print("Ploting Policy with RBF experience replay")
sb.heatmap(q_grid,xticklabels=x_grid,yticklabels=th_grid,cbar=False)
plt.title("Policy with RBF experience replay")
plt.xlabel("X")
plt.ylabel("Theta")
plt.show()
        
        
