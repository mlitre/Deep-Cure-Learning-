import gym
import numpy as np
from envs.deep_cure_env import DeepCure, random_base_infect_rate, random_lifetime, ForeignCountry
import matplotlib.pyplot as plt
from plotting import plot
from stable_baselines3 import DQN, A2C

env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], use_discrete = True, save_history=True)
env2 = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True)

# policy_kwargs = dict(net_arch=[6])
# model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=100000, log_interval=4)
# model.save("dqn_stable")

model2 = A2C("MlpPolicy", env2, verbose=1)
model2.learn(total_timesteps=100000)
model2.save("a2c_stable")

# del model # remove to demonstrate saving and loading

model = A2C.load("a2c_stable")

obs = env.reset(rate=2.7)
while True:
    # action, _states = model.predict(obs, deterministic=True)
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    if done:
      break

plot(env)