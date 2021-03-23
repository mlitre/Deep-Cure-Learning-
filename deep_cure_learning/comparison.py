from q_table_agent import greedy_policy, discretize
from reinforce_agent import logistic_regression
from saes_agent import NeuralNetworkPolicy
from envs.deep_cure_env import DeepCure, ForeignCountry, random_base_infect_rate, random_lifetime, random_delay
from plotting import plot
from stable_baselines3 import DQN
from prettytable import PrettyTable

import numpy as np

def constant_action(env, action,rate, lifetime, delay):
    state = env.reset(rate, lifetime, delay)
    end = False
    while not end:
        state, reward, end, _ = env.step(action)
    return env

def reinforce(env, theta, rate, lifetime, delay):
    obs = env.reset(rate,lifetime,delay)
    done = False
    while not done:
        probs = logistic_regression(obs, theta)
        actions = probs >= 0.5
        obs, reward, done, _ = env.step(actions)
    return env

def q_table(env, table, stepsize, num_states, rate, lifetime, delay):
    state = discretize(env.reset(rate,lifetime,delay), stepsize, num_states)
    end = False
    t = 0
    while not end :
        action = greedy_policy(state, table)
        state, reward, end, _ = env.step(action)
        state = discretize(state, stepsize, num_states)
    return env

def saes(env, policy, theta, rate, lifetime, delay):
    obs = env.reset(rate,lifetime,delay)
    done = False
    while not done:
        probs = policy(obs, theta)
        actions = probs >= 0.5
        obs, reward, done, _ = env.step(actions)
    return env

def stable(env, model, rate, lifetime, delay):
    obs = env.reset(rate, lifetime, delay)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    return env

def save_metrics(env, metric_dict, n):
    reward = sum(env.hist_reward)
    actions = np.array(env.hist_action)
    actions = np.sum(actions, axis=0)/len(env.hist_action)
    actions[2] = 1-actions[2]
    actions /= n
    dead_ratio =  env.hist_dead[-1]/env.population
    infected_ratio = env.hist_infected[-1]/env.population
    metric_dict['reward'].append(reward)
    metric_dict['actions'] += actions
    metric_dict['dead'].append(dead_ratio)
    metric_dict['infected'].append(infected_ratio)
    return reward

def compare(agents, n = 250):
    results = [{'reward': [], 'actions': np.zeros(3), 'dead': [], 'infected': [], 'best': 0} for i in range(len(agents))]
    current_reward = np.zeros(len(agents))
    for _ in range(n):
        rate = random_base_infect_rate()
        lifetime = random_lifetime()
        delay = [random_delay()]
        for i,(name,agent) in enumerate(agents):
            # run agent
            env = agent(rate, lifetime, delay)
            current_reward[i] = save_metrics(env, results[i], n)
        # get best reward
        results[np.argmax(current_reward)]['best'] += 1./n

    # print statistic
    table = PrettyTable()
    table.field_names = ['Agent', 'Best', 'Mean Reward', 'Std Reward', 'Mean Dead', 'Std Dead', 'Mean Infected', 'Std Infected', 'Actions']
    for i,(name,agent) in enumerate(agents):
        result = results[i]
        reward = np.array(result['reward'])
        dead = np.array(result['dead'])
        infected = np.array(result['infected'])
        masks, curfew, borders = result['actions']
        table.add_row([name, result['best'], np.mean(reward), np.std(reward), np.mean(dead), np.std(dead), np.mean(infected), np.std(infected), f'{masks} / {curfew} / {borders}'])

    print(table)


SEED = 22
np.random.seed(SEED)
env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed=SEED)
env.reset()

# this environment is for DQN which uses discrete action space
env2 = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], use_discrete = True, save_history=True, seed=SEED)
env2.reset()


theta = np.load('theta.npy')

q_table100 = np.load('qtable-100.npy')
q_table1000 = np.load('qtable-1000.npy')

theta_saes = np.load('saes-theta.npy')
policy = NeuralNetworkPolicy(env, one_layer=True)

theta_saes2 = np.load('saes-theta2.npy')
policy2 = NeuralNetworkPolicy(env, h_size=10, one_layer=False)

model = DQN.load("best_model")

agents = [
    ('Action nothing', lambda r,l,d: constant_action(env, [False,False,True], r, l, d)),
    ('Action nothing (closed borders)', lambda r,l,d: constant_action(env,[False,False,False], r, l, d)),
    ('Action masks', lambda r,l,d: constant_action(env, [True,False,True], r, l, d)),
    ('Action masks (closed borders)', lambda r,l,d: constant_action(env, [True,False,False], r, l, d)),
    ('Action curfew', lambda r,l,d: constant_action(env, [False,True,True], r, l, d)),
    ('Action curfew (closed borders)', lambda r,l,d: constant_action(env, [False,True,False], r, l, d)),
    ('Action both', lambda r,l,d: constant_action(env, [True,True,True], r, l, d)),
    ('Action both (closed borders)', lambda r,l,d: constant_action(env, [True,True,False], r, l, d)),
    ('reinforce', lambda r,l,d: reinforce(env, theta, r, l, d)),
    ('qtable 100', lambda r,l,d: q_table(env, q_table100, 100, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int), r, l, d)),
    ('qtable 1000', lambda r,l,d: q_table(env, q_table1000, 1000, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int), r, l, d)),
    ('SAES 1', lambda r,l,d: saes(env, policy, theta_saes, r, l, d)),
    ('SAES 2', lambda r,l,d: saes(env, policy2, theta_saes2, r, l, d)),
    ('DQN', lambda r,l,d: stable(env2, model, r, l, d))
]


# runs 250 environments and tests each agent
compare(agents)

# runs q_table agent
# q_table(env, q_table100, 100, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int), 1.7, 100, [40])

# runs saes agent
# saes(env, policy, theta_saes, 1.7, 100, [40])

# runs deep_q agent
# deep_q(env, theta, 1.7, 100, [40])

# runs a baseline agent
# constant_action(env, [True, True, False], 1.7, 100, [40])

# uncomment to plot the latest run
# print(f'Reward {sum(env.hist_reward)}')
# plot(env)
