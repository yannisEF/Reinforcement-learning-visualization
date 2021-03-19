# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse
from progress.bar import Bar

import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from savedVignette import SavedVignette
from slowBar import SlowBar
from vector_util import *

# To test (~8 minutes computing time)
# python3 Vignette.py --env Pendulum-v0 --inputDir Models/Pendulum --min_iter 8000 --max_iter 8000 --step_maxiter 500 --eval_maxiter 5 --nb_lines 10
# /!\ Should be used with caution as savedVignette can be very heavy /!\

if __name__ == "__main__":

	print("Parsing arguments")
	parser = argparse.ArgumentParser()

	# Model parameters
	parser.add_argument('--env', default='Pendulum-v0', type=str)# the environment to load
	parser.add_argument('--policy', default = 'MlpPolicy', type=str) # Policy of the model
	parser.add_argument('--tau', default=0.005, type=float) # the soft update coefficient (“Polyak update”, between 0 and 1)
	parser.add_argument('--gamma', default=1, type=float) # the discount model
	parser.add_argument('--learning_rate', default=0.0003, type=float) #learning rate for adam optimizer, the same learning rate will be used
																 # for all networks (Q-Values, model and Value function) it can be a function
																 #  of the current progress remaining (from 1 to 0)
	
	# Tools parameters
	parser.add_argument('--nb_lines', default=60, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : large 100, around model 10
	parser.add_argument('--stepalpha', default=0.25, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
	parser.add_argument('--eval_maxiter', default=5, type=float)# number of steps for the evaluation. Depends on environment.
	#	2D plot parameters
	parser.add_argument('--min_colormap', default=-1000, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--max_colormap', default=0, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--resolution', default=10, type=int)# the size of each pixel in 2D Vignette
	#	3D plot parameters
	parser.add_argument('--x_diff', default=2., type=float)# the space between each point along the x-axis
	parser.add_argument('--y_diff', default=2., type=float)# the space between each point along the y-axis
	parser.add_argument('--line_width', default=1., type=float)# the width of each line
	
	# File management
	#	Input parameters
	parser.add_argument('--inputDir', default="Models", type=str)# name of the directory containing the models to load
	parser.add_argument('--basename', default="rl_model_", type=str)# file prefix for the loaded model
	parser.add_argument('--min_iter', default=1, type=int)# iteration (file suffix) of the first model
	parser.add_argument('--max_iter', default=10, type=int)# iteration (file suffix) of the last model
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive models
	# 		Input policies parameters
	parser.add_argument('--policiesPath', default=None, type=str) # path to a list of policies to be included in Vignette
	#	Output parameters
	parser.add_argument('--saveInFile', default=True, type=bool)# true if want to save the savedVignette
	parser.add_argument('--save2D', default=True, type=bool)# true if want to save the 2D Vignette
	parser.add_argument('--save3D', default=True, type=bool)# true if want to save the 3D Vignette
	parser.add_argument('--directoryFile', default="SavedVignette", type=str)# name of the directory that will contain the vignettes
	parser.add_argument('--directory2D', default="Vignette_output", type=str)# name of the directory that will contain the 2D vignette
	parser.add_argument('--directory3D', default="Vignette_output", type=str)# name of the directory that will contain the 3D vignette
	args = parser.parse_args()


	# Creating environment and initialising model and parameters
	print("Creating environment\n")
	env = gym.make(args.env)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])
	
	# Instantiating the model
	model = SAC(args.policy, args.env,
				learning_rate=args.learning_rate,
				tau=args.tau,
				gamma=args.gamma)
	theta0 = model.policy.parameters_to_vector()
	num_params = len(theta0)
	
	# Retrieving the provided policies
	if args.policiesPath is not None:
		with open(args.policiesPath, 'rb') as handle:
			policies = pickle.load(handle)

	print('\n')
	
	# Plotting parameters
	v_min_fit = args.min_colormap
	v_max_fit = args.max_colormap

	# Choosing directions to follow
	D = getDirectionsMuller(args.nb_lines,num_params)

	# Name of the model files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i)+'_steps' for i in range(args.min_iter,
														args.max_iter+args.step_iter,
														args.step_iter)]

	# Compute fitness over these directions :
	previous_theta = None # Remembers theta
	for indice_file in range(len(filename_list)):
			
		# Change which model to load
		filename = filename_list[indice_file]

		# Load the model
		print("\nSTARTING : "+str(filename))
		model = SAC.load("{}/{}".format(args.inputDir, filename))
		
		# Get the new parameters
		theta0 = model.policy.parameters_to_vector()
		base_vect = theta0 if previous_theta is None else theta0 - previous_theta
		previous_theta = theta0
		print("Loaded parameters from file")

		# Processing the provided policies
		policyDistance, policyDirection = {}, [] # policyDistance has to be unordered as the directions will be shuffled
		if args.policiesPath is not None:
			with SlowBar('Computing the directions to input policies', max=len(policies)) as bar:
				for p in policies:
					distance = euclidienne(base_vect, p);	direction = (p - base_vect) / distance
					# Storing the directions to remove them from those already sampled
					policyDirection.append(direction)	
					# Storing the distances to the model in order to find a max alpha range
					# List not hashable
					policyDistance[direction] = distance
					# 	Remove the closest direction in those sampled
					del D[np.argmin([euclidienne(direction, dirK) for dirK in D])]

		# 	Adding the provided policies
		D += policyDirection
		# 	Ordering the directions
		D = order_all_by_proximity(D)

		# Evaluate the Model : mean, std
		print("Evaluating the model...")
		init_score = evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)[0]
		print("Model initial fitness : "+str(init_score))

		# Study the geometry around the model
		print("Starting study around the model...")
		theta_plus_scores, theta_minus_scores = [], []
		image, base_image = [], []

		#	Norm of the model
		length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
		# 		Direction taken by the model (normalized)
		d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist

		# Iterating over all directions, -1 is the direction that was initially taken by the model
		newVignette = SavedVignette(d, D, length_dist, policyDistance=policyDistance,
									v_min_fit=v_min_fit, v_max_fit=v_max_fit, stepalpha=args.stepalpha, resolution=args.resolution,
									x_diff=args.x_diff, y_diff=args.y_diff, line_width=args.line_width)
		for step in range(-1,len(D)):
			print("\nDirection ", step, "/", len(D)-1)
			# New parameters following the direction
			#	Changing the range and step of the Vignette if the optional input policies are beyond that range
			min_dist, max_dist = (args.minalpha, args.maxalpha) if args.policiesPath is None \
							else (args.minalpha, max(max(policyDistance.values()), args.maxalpha))
			step_dist = args.stepalpha * (max_dist - min_dist) / (args.maxalpha - args.minalpha)
			# 	Sampling new models' parameters following the direction
			theta_plus, theta_minus = getPointsDirection(theta0, num_params, min_dist, max_dist, step_dist, d)

			# Get the next direction
			if step != -1:	d = D[step]

			# Evaluate using new parameters
			scores_plus, scores_minus = [], []
			with SlowBar('Evaluating along the direction', max=len(theta_plus)) as bar:
				for param_i in range(len(theta_plus)):
					# 	Go forward in the direction
					model.policy.load_from_vector(theta_plus[param_i])
					#		Get the new performance
					scores_plus.append(evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)[0])
					# 	Go backward in the direction
					model.policy.load_from_vector(theta_minus[param_i])
					#		Get the new performance
					scores_minus.append(evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)[0])
					
					bar.next()

			# Inverting scores for a symetrical Vignette (theta_minus going left, theta_plus going right)
			scores_minus = scores_minus[::-1]
			line = scores_minus + [init_score] + scores_plus
			# 	Adding the line to the image
			if step == -1:	newVignette.baseLines.append(line)
			else:	newVignette.lines.append(line)

		try:
			# Computing the 2D Vignette
			if args.save2D is True:	newVignette.plot2D()
			# Computing the 3D Vignette
			if args.save3D is True: newVignette.plot3D()
		except Exception as e:
			newVignette.saveInFile("{}/temp/{}.xz".format(args.directoryFile, filename))
			e.print_exc()

		
		# Saving the Vignette
		angles3D = [20,45,50,65] # angles at which to save the plot3D
		elevs= [0, 30, 60]
		newVignette.saveAll(filename, saveInFile=args.saveInFile, save2D=args.save2D, save3D=args.save3D,
							directoryFile=args.directoryFile, directory2D=args.directory2D, directory3D=args.directory3D,
							angles3D=angles3D, elevs=elevs)
	

	env.close()
