from envs.deep_cure_env import DeepCure, random_base_infect_rate, random_lifetime, ForeignCountry
from plotting import plot
import gym
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(s, theta):
    # s: state space
    # theta action space x state space +1
    # result action space
    a = np.zeros((len(s)+1,1))
    a[:len(s),0] = s
    a[len(s),0] = 1
    prob_active = sigmoid(theta @ a)
    return prob_active.reshape(-1)

def draw_action(s, theta):
    # theta : dim action space x state space
    probs = logistic_regression(s, theta)

    actions = np.random.rand(probs.shape[0]) < probs
    return actions

# Generate an episode
def play_one_episode(env, theta, rate = None, lifetime = None):
    s_t = env.reset(rate=rate, lifetime=lifetime)

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    done = False
    while not done:
        a_t = draw_action(s_t, theta)
        s_t, r_t, done, info = env.step(a_t)

        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

    return episode_states, episode_actions, episode_rewards, env.v_base_infect_rate

def score_on_multiple_episodes(env, theta, rate = None, lifetime = None, num_episodes=10):
    average_return = 0
    for episode_index in range(num_episodes):
        _, _, episode_rewards, _ = play_one_episode(env, theta, rate, lifetime)
        total_rewards = sum(episode_rewards)
        average_return += (1.0 / num_episodes) * total_rewards
    return average_return

def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta, rate):
    H = len(episode_rewards)
    PG = np.zeros_like(theta)
    for t in range(H):
        probs = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])
        # episode_states[t] : observations at time t [previous new infected, new infected, 1]
        states = np.ones(len(episode_states[t])+1)
        states[:len(episode_states[t])] = episode_states[t]
        for i,prob in enumerate(probs):
            if not a_t[i]:
                g_theta_log_pi = -(1-prob) * states * R_t
            else:
                g_theta_log_pi = prob * states * R_t
            PG[i] += g_theta_log_pi

    return PG

def train(env, theta_init, alpha_init = 0.001, iterations = 2000, rate = None, lifetime = None):

    theta = theta_init
    average_returns = []

    R = score_on_multiple_episodes(env, theta, rate, lifetime)
    average_returns.append(R)

    # Train for iterations
    for i in range(iterations):
        episode_states, episode_actions, episode_rewards, played_rate = play_one_episode(env, theta, rate, lifetime)
        alpha = alpha_init /np.sqrt(1+i)
        PG = compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta, played_rate)
        old_theta = np.copy(theta)
        theta = theta + alpha * PG
        R = score_on_multiple_episodes(env, theta, rate, lifetime)
        average_returns.append(R)

    return theta, average_returns

if __name__ == "__main__":

    SEED = 42

    np.random.seed(SEED)

    env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed=SEED)
    env.reset()

    # observation x action +1
    theta = np.zeros((env.observation_space.shape[0],env.action_space.n + 1))

    theta, average_returns = train(env, theta)
    print(theta)
    np.save('theta.npy', theta)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('iterations')
    ax.set_ylabel('average reward')
    ax.plot(range(len(average_returns)), average_returns)
    fig.tight_layout()
    plt.show()
