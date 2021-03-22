import gym
import numpy as np
from envs.deep_cure_env import DeepCure, random_base_infect_rate, random_lifetime, ForeignCountry
import matplotlib.pyplot as plt
from plotting import plot
from stable_baselines3 import DQN, A2C
import torch as th 
from stable_baselines3.common.callbacks import EvalCallback

def lr(progress):
  return 0.001*np.sqrt(progress/100)

env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], use_discrete = True, save_history=True)
eval_callback = EvalCallback(env, best_model_save_path='./',
                             log_path='./', eval_freq=500,
                             deterministic=True, render=False)

policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=[5])
model = DQN("MlpPolicy", env, batch_size=2, learning_rate=lr, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000, n_eval_episodes=100000, callback=eval_callback)
model.save("dqn_stable")

model = DQN.load("dqn_stable")


obs = env.reset(rate=2.7)
while True:
    # action, _states = model.predict(obs, deterministic=True)
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    if done:
      break

plot(env)