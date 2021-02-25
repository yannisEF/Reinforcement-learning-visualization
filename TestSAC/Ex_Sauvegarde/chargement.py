import gym
import numpy as np

from stable_baselines3 import SAC

toLoad = 5

# Example on how to load an agent, then simulate the trained agent
env = gym.make("Swimmer-v2")
model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)

# Load the agent
model.load("Saves/save"+str(toLoad))

# Simulate the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()