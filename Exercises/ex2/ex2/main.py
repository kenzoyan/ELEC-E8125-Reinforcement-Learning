# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


def values_iterations(env,gamma=1.0,itNum=100):
    
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    
    old_values=value_est.copy()
    old_policy=policy.copy()

    for it in range(itNum):
        # iterate every state in 2D
        env.clear_text()
        for w in range(env.w):
            for h in range(env.h):
                Q_value=[]
                for tran in env.transitions[w,h]:
                    # print("tran:")
                    # print(tran)
                    A_value=0
                    for next_state,reward,done,prob in tran:
                        # consider all situation under different prob
                        A_value+=prob*(reward+(gamma*value_est[next_state] if not done else 0))
                    Q_value.append(A_value)
                # print(it,Q_value)
                value_est[w,h]=np.max(Q_value)
                policy[w,h]=np.argmax(Q_value)
        
        # value difference checks converge

        # if ((value_est-old_values)<epsilon).all():
        #     print("converge at "+ str(it))
        #     break
        # else:
        #     # print("Not converge at "+ str(it))
        #     old_values=np.copy(value_est)           # should use np.copy, directly equal causes fault
        

        # policy difference checks converge

        # if ((policy-old_policy)<epsilon).all():
        #     print("Policy converge at "+ str(it))
        #     break
        # else:
        #     print("Policy didn't converge at "+ str(it))
        #     old_policy=np.copy(policy)           # should use np.copy, directly equal causes fault
        
        # print(str(it)+"-----------------------")
        # print(value_est)
        
        
    
    return value_est,policy
        




if __name__ == "__main__":
    # Reset the environment
    state = env.reset()
    gamma=0.9
    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    value_est, policy=values_iterations(env,gamma,100)
    
    
    # Show the values and the policy
    print(value_est)
    print(policy)
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(10)

    # value function converge at 29
    # policy function converge at 23

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    # fnames = "values-task3.npy", "policy-task3.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Load saved the state values and the policy
    value_est=np.load("values.npy")
    policy=np.load("policy.npy")
    print(value_est)
    print(policy)
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(10)
    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    
    results=[]
    for ep in range(1000):
        discount_return=0
        i=0
        done = False
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            action = policy[state]

            # Step the environment
            state, reward, done, _ = env.step(action)
            discount_return+=reward*(gamma**i)
            i+=1
            # Render and sleep
            # env.render()
            # sleep(0.2)
        
        print("Episode"+ str(ep)+": "+ str(round(discount_return,5)))
        results.append(discount_return)

        state = env.reset()

    # print(results)
    print("--------Average& STD") # 0.683 & 1.361
    print(np.average(results))
    print(np.std(results))

