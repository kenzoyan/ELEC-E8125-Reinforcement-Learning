import gym
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import seaborn as sb
import copy
np.random.seed(123)

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')

env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16
# x_min, x_max = -2.4, 2.4
# v_min, v_max = -3, 3
# th_min, th_max = -0.3, 0.3
# av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

l_x_min, l_x_max = -1.2, 1.2
l_y_min, l_y_max = -0.3, 1.2
l_xdot_min, l_xdot_max = -2.4, 2.4
l_ydot_min, l_ydot_max = -2.0, 2.0
l_th_min, l_th_max = -6.28,6.28
l_thdot_min, l_thdot_max = -8, 8
l_cl_max, l_cl_min=0,1
l_cr_max, l_cr_min=0,1




# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
# for 0.1 a=2222
a = 2222  # TODO: Set the correct value.

initial_q = 0  # T3: Set to 50

# Create discretization grid
# x_grid = np.linspace(x_min, x_max, discr)
# v_grid = np.linspace(v_min, v_max, discr)
# th_grid = np.linspace(th_min, th_max, discr)
# av_grid = np.linspace(av_min, av_max, discr)

# q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

l_x_grid = np.linspace(l_x_min, l_x_max, discr)
l_y_grid = np.linspace(l_y_min, l_y_max, discr)
l_xdot_grid = np.linspace(l_xdot_min, l_xdot_max, discr)
l_ydot_grid = np.linspace(l_ydot_min, l_ydot_max, discr)
l_th_grid = np.linspace(l_th_min, l_th_max, discr)
l_thdot_grid = np.linspace(l_thdot_min, l_thdot_max, discr)
l_cl_grid = np.linspace(l_cl_max, l_cl_min, 2) # binary choices
l_cr_grid = np.linspace(l_cr_max, l_cr_min, 2)


q_grid = np.zeros((discr, discr, discr, discr,discr, discr, 2 , 2 ,num_of_actions)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


# def get_cell_index(state):
#     x = find_nearest(x_grid, state[0])
#     v = find_nearest(v_grid, state[1])
#     th = find_nearest(th_grid, state[2])
#     av = find_nearest(av_grid, state[3])
#     return x, v, th, av

def get_cell_index_lunar(state):
    x = find_nearest(l_x_grid, state[0])
    y = find_nearest(l_y_grid, state[1])
    xdot=find_nearest(l_xdot_grid,state[2])
    ydot=find_nearest(l_ydot_grid,state[3])
    th = find_nearest(l_th_grid, state[4])
    thdot = find_nearest(l_thdot_grid, state[5])
    cl=find_nearest(l_cr_grid,state[6])
    cr=find_nearest(l_cr_grid,state[7])    
    
    return x, y, xdot, ydot, th, thdot , cl , cr


def get_action(state, q_values, epsilon, greedy=False):
    # TODO: Implement epsilon-greedy

    if random.uniform(0,1)<epsilon:
        # random action
        return env.action_space.sample() # 0/1/2/3

    else:
        # Choose current best action
        
        return np.argmax(q_values[get_cell_index_lunar(state)])


    # raise NotImplementedError("Implement epsilon-greedy")


def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    old_cell_index = get_cell_index_lunar(old_state)
    new_cell_index = get_cell_index_lunar(new_state)
    
    # print("new_cell_index",new_cell_index)
    # print(q_array.shape, env.action_space)
    
    q_values=[q_array[new_cell_index+ (a,)] for a in range(env.action_space.n)]
    
    # print(q_values)

    q_max=max(q_values)
    # print(q_max)
    q_array[old_cell_index+(action,)]+=alpha*(reward + gamma * q_max - q_array[old_cell_index+(action,)])
    # print("test:", q_array[old_cell_index+(action,)])
    # raise NotImplementedError("Implement Q-value update")

# Training loop
ep_lengths, epl_avg = [], []
total_reward, total_reward_avg=[],[]
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    # epsilon = 0  # T1: GLIE/constant, T3: Set to 0
    epsilon=np.round(a/(a+ep),2)
    current_reward=0
    while not done:
        action = get_action(state, q_grid, epsilon,greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
            

        else:
            env.render()
        state = new_state
        steps += 1
        current_reward+=reward
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))

    total_reward.append(current_reward)
    total_reward_avg.append(np.mean(total_reward[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}, epsilon:{}, avg reward: {:.2f} ".format(ep, np.mean(ep_lengths[max(0, ep-200):]),epsilon,np.mean(total_reward[max(0, ep-200):])))


# print("q_grid",np.sum(q_grid))
# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.amax(q_grid,axis=4) # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib

# sb.heatmap(np.mean(values, axis=(1, 3)))
# plt.show()

# Draw plots timesteps
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

# Draw plots reward
plt.plot(total_reward)
plt.plot(total_reward_avg)
plt.legend(["Episode Reward", "500 episode average"])
plt.title("Episode Rewards")
plt.show()

# Q是表格函数，对简单的有限状态任务有效，如果是连续空间，或者复杂离散空间，都没法用查表的方法得到某个状态的Q值，应该用神经网络
