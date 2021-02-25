import gym
import numpy as np

from stable_baselines3 import SAC

# Example on how to save an agent
env = gym.make("Swimmer-v2")
model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)

nb_train = 10
for k in range(nb_train):
    # Train the agent
    model.learn(total_timesteps=int(5000))

    # Save the agent
    model.save("Saves/save"+str(k+1))

del model