from envs.deep_cure_env import DeepCure, ForeignCountry, random_base_infect_rate
from plotting import plot
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

def discretize(state, stepsize, num_states):
    return np.minimum(state / stepsize, num_states - 1).astype(int)

def index(state, action = None):
    if action is None:
        # corresponds to [:]
        action = slice(None,None)
    else:
        action = sum([b * (2 ** i) for i,b in enumerate(action)])
    index = tuple((*state,action))
    return index

def greedy_policy(state, q_array):
    action_index = np.argmax(q_array[index(state)])
    action = np.array([int((action_index / (2 ** i)) % 2) for i in range(int(math.log2(q_array.shape[-1])))])
    return action

def epsilon_greedy_policy(env, state, q_array, epsilon):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_policy(state, q_array)
    return action

def q_learning(environment, alpha=0.1, alpha_factor=0.9995, gamma=0.99, epsilon=0.5, num_episodes=10000, rate = None, stepsize = 20, max_steps = 100):
    q_array_history = [0]
    last_q_array = None
    alpha_history = []
    num_states = np.minimum((environment.observation_space.high - environment.observation_space.low)/stepsize, max_steps).astype(int)
    num_actions = 2**environment.action_space.n
    q_array = np.zeros(list(num_states) + [num_actions])   # Initial Q table
    for episode_index in range(num_episodes):
        alpha_history.append(alpha)

        # Update alpha
        if alpha_factor is not None:
            alpha = alpha * alpha_factor

        is_final_state = False
        state = discretize(environment.reset(rate), stepsize, num_states)

        while not is_final_state:
            action = epsilon_greedy_policy(environment, state, q_array, epsilon)
            new_state, reward, is_final_state, info = environment.step(action)

            new_state = discretize(new_state, stepsize, num_states)
            new_action = greedy_policy(new_state, q_array)
            q_array[index(state, action)] = q_array[index(state, action)] + alpha * (reward + gamma * q_array[index(new_state, new_action)] - q_array[index(state, action)])

            state = new_state

        if last_q_array is not None:
           q_array_history.append(np.max(np.absolute(q_array - last_q_array)))
        last_q_array = q_array.copy()

    return q_array, q_array_history, alpha_history

def hyperparameter_search(stepsize, max_steps, gamma_range):
    def eval_q_table(env, q_table, n=100):
        rewards = np.zeros(n)
        for i in range(n):
            state = discretize(env.reset(), stepsize, num_states)
            done = False
            while not done:
                action = greedy_policy(state, q_table)
                state, reward, done, _ = env.step(action)
                state = discretize(state, stepsize, num_states)
            rewards[i] = sum(env.hist_reward)
        return np.mean(rewards)
    r = []
    for gamma in gamma_range:
        SEED = 42

        np.random.seed(SEED)

        env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed = SEED)
        num_states = np.minimum((env.observation_space.high - env.observation_space.low)/stepsize, max_steps).astype(int)
        q_table, _, _  = q_learning(env, stepsize=stepsize, max_steps=max_steps, gamma=gamma)
        reward = eval_q_table(env, q_table)
        r.append(reward)
        print(f'Gamma={gamma}: {reward}')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\gamma$')
    ax.set_ylabel('avg. reward')
    ax.plot(gamma_range , r)
    plt.show()

if __name__ == "__main__":

    #hyperparameter_search(100, 10, [i/100. for i in range(80,100)])
    
    SEED = 42

    np.random.seed(SEED)

    env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed = SEED)

    q_table, q_array_history, alpha_history = q_learning(env, epsilon=0.5, gamma = 0.98, stepsize = 100, max_steps = 10)
    np.save(f'qtable-100.npy', q_table)

    np.random.seed(SEED)

    q_table2, q_array_history2, _ = q_learning(env, epsilon=0.5, gamma = 0.98, stepsize = 1000, max_steps = 10)
    np.save(f'qtable-1000.npy', q_table2)

    fig = plt.figure()
    ax0 = fig.add_subplot(2,1,1)
    ax0.set_title('Q Table Convergence')
    ax0.set_xlabel('iterations')
    ax0.set_ylabel('absolute difference')
    ax0.plot(range(len(q_array_history)), q_array_history, label='q-table 100')
    ax0.plot(range(len(q_array_history2)), q_array_history2, label='q-table 1000')
    ax0.legend()

    ax1 = fig.add_subplot(2,1,2)
    ax1.set_title('$\\alpha$')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('$\\alpha$')
    ax1.plot(range(len(alpha_history)), alpha_history)

    fig.tight_layout()

    plt.show()
