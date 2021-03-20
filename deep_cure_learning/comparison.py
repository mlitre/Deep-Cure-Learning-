from q_table_agent import greedy_policy, discretize
from deep_q_agent import logistic_regression
from saes_agent import NeuralNetworkPolicy
from envs.deep_cure_env import DeepCure, ForeignCountry, random_base_infect_rate, random_lifetime, random_delay
from plotting import plot
from stable_baselines3 import DQN

import numpy as np

def action_ratio(env):
    actions = np.array(env.hist_action)
    return np.sum(actions, axis=0)/len(env.hist_action)

def constant_action(action,rate, lifetime, delay):
    state = env.reset(rate, lifetime, delay)
    end = False
    while not end :
        state, reward, end, _ = env.step(action)
    return sum(env.hist_reward), action_ratio(env)

def deep_q(env, theta, rate, lifetime, delay):
    obs = env.reset(rate,lifetime,delay)
    done = False
    while not done:
        probs = logistic_regression(obs, theta)
        actions = probs >= 0.5
        obs, reward, done, _ = env.step(actions)
    return sum(env.hist_reward), action_ratio(env)

def q_table(env, table, stepsize, num_states, rate, lifetime, delay):
    state = discretize(env.reset(rate,lifetime,delay), stepsize, num_states)
    end = False
    t = 0
    while not end :
        action = greedy_policy(state, table)
        state, reward, end, _ = env.step(action)
        state = discretize(state, stepsize, num_states)
    return sum(env.hist_reward), action_ratio(env)

def saes(env, policy, theta, rate, lifetime, delay):
    obs = env.reset(rate,lifetime,delay)
    done = False
    while not done:
        probs = policy(obs, theta)
        actions = probs >= 0.5
        obs, reward, done, _ = env.step(actions)
    return sum(env.hist_reward), action_ratio(env)

def stable(env, model, rate, lifetime, delay):
    obs = env.reset(rate, lifetime, delay)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    return sum(env.hist_reward), action_ratio(env)

def tweak_params(env, n=100):
    prewards = [0] * 8
    prewards_hist = list()
    # pandemic
    for i in range(n):
        rate = random_base_infect_rate()
        while rate <= 1.:
            rate = random_base_infect_rate()
        lifetime =30
        delay = [0]
        prewards[0], _ = constant_action([False,False,False], rate, lifetime, delay)
        prewards[1], _ = constant_action([True,False,False], rate, lifetime, delay)
        prewards[2], _ = constant_action([False,True,False], rate, lifetime, delay)
        prewards[3], _ = constant_action([True,True,False], rate, lifetime, delay)
        prewards[4], _ = constant_action([False,False,True], rate, lifetime, delay)
        prewards[5], _ = constant_action([True,False,True], rate, lifetime, delay)
        prewards[6], _ = constant_action([False,True,True], rate, lifetime, delay)
        prewards[7], _ = constant_action([True,True,True], rate, lifetime, delay)
        prewards_hist.append(list(prewards))
    rewards = [0] * 8
    rewards_hist = list()
    # non-pandemic
    for i in range(n):
        rate = random_base_infect_rate()
        while rate > 1.:
            rate = random_base_infect_rate()
        lifetime = 30
        delay = [0]
        rewards[0], _ = constant_action([False,False,False], rate, lifetime, delay)
        rewards[1], _ = constant_action([True,False,False], rate, lifetime, delay)
        rewards[2], _ = constant_action([False,True,False], rate, lifetime, delay)
        rewards[3], _ = constant_action([True,True,False], rate, lifetime, delay)
        rewards[4], _ = constant_action([False,False,True], rate, lifetime, delay)
        rewards[5], _ = constant_action([True,False,True], rate, lifetime, delay)
        rewards[6], _ = constant_action([False,True,True], rate, lifetime, delay)
        rewards[7], _ = constant_action([True,True,True], rate, lifetime, delay)
        rewards_hist.append(list(rewards))
    pr = np.mean(np.array(prewards_hist), axis=0)
    r = np.mean(np.array(rewards_hist), axis=0)
    print(f'No Action borders closed:{pr[0]} {r[0]}')
    print(f'No Action borders open:{pr[4]} {r[4]}')
    print()
    print(f'Masks borders closed:{pr[1]} {r[1]}')
    print(f'Masks borders open:{pr[5]} {r[5]}')
    print()
    print(f'Curfew borders closed:{pr[2]} {r[2]}')
    print(f'Curfew borders open:{pr[6]} {r[6]}')
    print()
    print(f'All borders closed:{pr[3]} {r[3]}')
    print(f'All borders open:{pr[7]} {r[7]}')


def compare(env, env2, theta, q_tables, policy, theta_saes, policy2, theta_saes2, model, n = 250):
    cnts = [0] * (12 + len(q_tables))
    rewards = [0] * (12 + len(q_tables))
    actions = [np.zeros((env.action_space.n,))] * (3+len(q_tables))
    reward_hist = list()
    for i in range(n):
        rate = random_base_infect_rate()
        lifetime = random_lifetime()
        delay = [random_delay()]
        rewards[0], _ = constant_action([False,False,False], rate, lifetime, delay)
        rewards[1], _ = constant_action([True,False,False], rate, lifetime, delay)
        rewards[2], _ = constant_action([False,True,False], rate, lifetime, delay)
        rewards[3], _ = constant_action([True,True,False], rate, lifetime, delay)
        rewards[4], _ = constant_action([False,False,True], rate, lifetime, delay)
        rewards[5], _ = constant_action([True,False,True], rate, lifetime, delay)
        rewards[6], _ = constant_action([False,True,True], rate, lifetime, delay)
        rewards[7], _ = constant_action([True,True,True], rate, lifetime, delay)
        rewards[8], deep_q_actions = deep_q(env, theta, rate, lifetime, delay)
        rewards[9], saes_actions = saes(env, policy, theta_saes, rate, lifetime, delay)
        rewards[10], saes2_actions = saes(env, policy2, theta_saes2, rate, lifetime, delay)
        rewards[11], _ = stable(env2, model, rate, lifetime, delay)
        actions[0] = actions[0] + deep_q_actions
        actions[1] = actions[1] + saes_actions
        actions[2] = actions[2] + saes2_actions
        for i,(table,stepsize,num_states) in enumerate(q_tables):
            rewards[12+i], q_table_actions = q_table(env, table, stepsize, num_states, rate, lifetime, delay)
            actions[3+i] = actions[3+i] + q_table_actions

        reward_hist.append(list(rewards))
        cnts[np.argmax(rewards)] += 1

    actions = np.array(actions)
    actions *= 1./n
    actions[:,2] = 1 - actions[:,2]
    cnts = np.array(cnts, dtype=float)
    cnts *= 1./n
    reward_hist = np.array(reward_hist)
    print(f'Action Nothing : {cnts[0]}')
    print(f'Action Masks : {cnts[1]}')
    print(f'Action Curfew : {cnts[2]}')
    print(f'Action All : {cnts[3]}')
    print(f'Action Nothing (open border) : {cnts[4]}')
    print(f'Action Masks (open border): {cnts[5]}')
    print(f'Action Curfew (open border): {cnts[6]}')
    print(f'Action All (open border): {cnts[7]}')
    print(f'Action deep-q agent: {cnts[8]}')
    print(f'Action saes agent (l1): {cnts[9]}')
    print(f'Action saes agent (l2): {cnts[10]}')
    print(f'Action DQN Stable agent: {cnts[11]}')
    for i,(_,stepsize,_) in enumerate(q_tables):
        print(f'Action q_table {stepsize}: {cnts[12+i]}')
    print()
    print(f'Action nothing\t\t{np.mean(reward_hist[:,0])} ({np.std(reward_hist[:,0])})')
    print(f'Action masks\t\t{np.mean(reward_hist[:,1])} ({np.std(reward_hist[:,1])})')
    print(f'Action curfew\t\t{np.mean(reward_hist[:,2])} ({np.std(reward_hist[:,2])})')
    print(f'Action all\t\t{np.mean(reward_hist[:,3])} ({np.std(reward_hist[:,3])})')
    print(f'Action nothing (open border)\t\t{np.mean(reward_hist[:,4])} ({np.std(reward_hist[:,4])})')
    print(f'Action masks (open border)\t\t{np.mean(reward_hist[:,5])} ({np.std(reward_hist[:,5])})')
    print(f'Action curfew (open border)\t\t{np.mean(reward_hist[:,6])} ({np.std(reward_hist[:,6])})')
    print(f'Action all (open border)\t\t{np.mean(reward_hist[:,7])} ({np.std(reward_hist[:,7])})')
    print(f'Action deep-q agent\t\t{np.mean(reward_hist[:,8])} ({np.std(reward_hist[:,8])})')
    print(f'Action saes agent (1 layer)\t\t{np.mean(reward_hist[:,9])} ({np.std(reward_hist[:,9])})')
    print(f'Action saes agent (2 layer)\t\t{np.mean(reward_hist[:,10])} ({np.std(reward_hist[:,10])})')
    print(f'Action DQN Stable \t\t{np.mean(reward_hist[:,11])} ({np.std(reward_hist[:,11])})')
    for i,(_,stepsize,_) in enumerate(q_tables):
        print(f'Action q_table {stepsize}:\t\t{np.mean(reward_hist[:,12+i])} ({np.std(reward_hist[:,12+i])}')
    print()
    print(f'Deep-q actions: {actions[0]}')
    print(f'SAES actions: {actions[1]}')
    print(f'SAES2 actions: {actions[2]}')
    for i,(_,stepsize,_) in enumerate(q_tables):
        print(f'q_table {stepsize}: {actions[3+i]}')

SEED = 22
np.random.seed(SEED)
env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed=SEED)
env.reset()

env2 = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], use_discrete = True, save_history=True, seed=SEED)
env2.reset()


theta = np.load('theta.npy')

q_tables = [
    (np.load('qtable-100.npy'), 100, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int)),
    (np.load('qtable-1000.npy'), 1000, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int))
]

theta_saes = np.load('saes-theta.npy')
policy = NeuralNetworkPolicy(env, one_layer=True)

theta_saes2 = np.load('saes-theta2.npy')
policy2 = NeuralNetworkPolicy(env, h_size=10, one_layer=False)

model = DQN.load("dqn_stable")

# runs 250 environments and tests each agent
compare(env, env2, theta, q_tables, policy, theta_saes, policy2, theta_saes2, model)



# runs q_table agent
# q_table(env, q_tables[1][0], q_tables[1][1], q_tables[1][2], 1.7, 100, [40])

# runs saes agent
# saes(env, policy, theta_saes, 1.7, 100, [40])

# runs deep_q agent
# deep_q(env, theta, 1.7, 100, [40])

# runs a baseline agent
# constant_action([True, True, False], 1.7, 100, [40])

# uncomment to plot the latest run
# print(f'Reward {sum(env.hist_reward)}')
# plot(env)
