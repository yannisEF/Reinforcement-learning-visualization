import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)
eval_env = gym.make("Swimmer-v2")


#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

#print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Train the agent
theta0 = model.get_parameters()
dictTheta = theta0["policy"]

for k in dictTheta: print(k + "\t len :",len(dictTheta[k]))
# Save the agent
#model.save("dqn_lunar")
#del model  # delete trained model to demonstrate loading