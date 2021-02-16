import gym
import numpy as np
from wrappers.perf_writer import PerfWriter
from visu.visu_results import plot_data
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)
eval_env = gym.make("Swimmer-v2")

perf=PerfWriter(eval_env)
perf.set_file_name("test3")
mean_reward, std_reward = evaluate_policy(model, perf, n_eval_episodes=10, deterministic=True)

plot_data('data/save/reward_test3.txt',"reward")
plt.show()
