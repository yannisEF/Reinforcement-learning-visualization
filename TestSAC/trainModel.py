import gym
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Saves a model's training process

if __name__ == "__main__":
    print("Parsing arguments")
    parser = argparse.ArgumentParser()

	# Model parameters
	parser.add_argument('--env', default='Swimmer-v2', type=str)
	parser.add_argument('--policy', default = 'MlpPolicy', type=str) # Policy of the model
	parser.add_argument('--tau', default=0.005, type=float) # the soft update coefficient (“Polyak update”, between 0 and 1)
	parser.add_argument('--gamma', default=0.99, type=float) # the discount fmodel
	parser.add_argument('--learning_rate', default=0.0003, type=float) #learning rate for adam optimizer, the same learning rate will be used
																 # for all networks (Q-Values, model and Value function) it can be a function
																 #  of the current progress remaining (from 1 to 0)
    
    # Save parameters
    parser.add_argument('--save_path', default='Models', type=str) # path to save
    parser.add_argument('--name_prefix', default='rl_model', type=str) # prefix of saves' name
    parser.add_argument('--save_freq', default=1000, type=int) # frequency of the save
    parser.add_argument('--max_learn', default=50000, type=int) # Number of steps to learning process
    args = parser.parse_args()

    # Creating environment and initialising model and parameters
    print("Creating environment\n")
    eval_env = gym.make(args.env)
	model = SAC(args.policy, args.env,
				learning_rate=args.learning_rate,
				tau=args.tau,
				gamma=args.gamma)

    # Use deterministic actions for evaluation
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.save_path,
                                            name_prefix=args.name_prefix)

    # Starting the learning process
    model.learn(args.max_learn, callback=checkpoint_callback)