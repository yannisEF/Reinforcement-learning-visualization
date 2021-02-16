import gym
import numpy as np
from BasicPolicyGradientLabsmaster.wrappers.PerfWriter import PerfWriter


from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)
eval_env = gym.make("Swimmer-v2")

perf=PerfWriter(eval_env)
perf.set_filename("test")
mean_reward, std_reward = evaluate_policy(model, p, n_eval_episodes=10, deterministic=True)
