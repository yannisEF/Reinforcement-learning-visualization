import argparse
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

print("Parsing arguments")
parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--env', default='Swimmer-v2', type=str)# the environment to load
parser.add_argument('--policy', default = 'MlpPolicy', type=str) # Policy of the model
parser.add_argument('--tau', default=0.005, type=float) # the soft update coefficient (“Polyak update”, between 0 and 1)
parser.add_argument('--gamma', default=0.99, type=float) # the discount model
parser.add_argument('--learning_rate', default=0.0003, type=float) #learning rate for adam optimizer, the same learning rate will be used
															 # for all networks (Q-Values, model and Value function) it can be a function
															 #  of the current progress remaining (from 1 to 0)
parser.add_argument('--eval_maxiter', default=1000, type=float)# number of steps for the evaluation. Depends on environment.

parser.add_argument('--min_iter', default=1000, type=int)# iteration (file suffix) of the first model
parser.add_argument('--max_iter', default=50000, type=int)# iteration (file suffix) of the last model
parser.add_argument('--step_iter', default=1000, type=int)# iteration step between two consecutive models


# File management
parser.add_argument('--directory', default="Models", type=str) # Dossier de chargement du modèle
parser.add_argument('--nameModel', default="rl_model_", type=str) # Nom du modèle à charger
args = parser.parse_args()

# Instantiating model
env = gym.make(args.env)
model = SAC(args.policy, args.env,
    		learning_rate=args.learning_rate,
			tau=args.tau,
			gamma=args.gamma)

# Name of the model files to analyse consecutively with the same set of directions: 
filename_list = [args.nameModel+str(i)+'_steps' for i in range(args.min_iter,
													args.max_iter+args.step_iter,
													args.step_iter)]

scores, variances = [], []
for filename in filename_list:
    # Loading the desired model
    print("\nSTARTING : "+filename)
    model.load("{}/{}".format(args.directory, filename))

    # Evaluate the Model : mean, std
    print("Evaluating the model...")
    init_score, variance = evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)
    scores.append(init_score)
    variances.append(variance)

plt.figure()
plt.plot(scores)

plt.figure()
plt.plot(variances)

plt.show()