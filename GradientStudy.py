# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse

from progress.bar import Bar
from PIL import Image
from PIL import ImageDraw
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import colorTest

from vector_util import *
from slowBar import SlowBar
from savedGradient import SavedGradient


# To test (~5 minutes computing time)
# python3 GradientStudy.py --env Pendulum-v0 --directory Models/Pendulum --min_iter 500 --max_iter 10000 --step_iter 500 --eval_maxiter 5

if __name__ == "__main__":

	print("Parsing arguments")
	parser = argparse.ArgumentParser()

	# Model parameters
	parser.add_argument('--env', default='Swimmer-v2', type=str)
	parser.add_argument('--policy', default = 'MlpPolicy', type=str) # Policy of the model
	parser.add_argument('--tau', default=0.005, type=float) # the soft update coefficient (“Polyak update”, between 0 and 1)
	parser.add_argument('--gamma', default=1, type=float) # the discount fmodel
	parser.add_argument('--learning_rate', default=0.0003, type=float) #learning rate for adam optimizer, the same learning rate will be used
																 # for all networks (Q-Values, model and Value function) it can be a function
																 #  of the current progress remaining (from 1 to 0)

	# Tools parameters
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : large 100, around model 10
	parser.add_argument('--stepalpha', default=0.25, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
	parser.add_argument('--eval_maxiter', default=5, type=float)# number of steps for the evaluation.
	#	Drawing parameters
	parser.add_argument('--pixelWidth', default=20, type=int)# width of each pixel
	parser.add_argument('--pixelHeight', default=10, type=int)# height of each pixel
	parser.add_argument('--maxValue', default=360, type=int)# max score value for colormap used (dependent of benchmark used)
	parser.add_argument('--line_height', default=3, type=int) # The height in number of pixel for each result
	#	Dot product parameters
	parser.add_argument('--dotWidth', default=150, type=int)# max width of the dot product (added on the side)
	parser.add_argument('--dotText', default=True, type=str)# true if want to show value of the dot product
	parser.add_argument('--xMargin', default=10, type=int) # xMargin for the side panel

	# File management
	parser.add_argument('--directory', default="Models", type=str)# name of the directory containing the models to load
	parser.add_argument('--basename', default="rl_model_", type=str)# file prefix for the loaded model
	parser.add_argument('--min_iter', default=1, type=int)# iteration (file suffix) of the first model
	parser.add_argument('--max_iter', default=10, type=int)# iteration (file suffix) of the last model
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive models
	#	Output parameters
	parser.add_argument('--saveFile', default=True, type=bool) # True if want to save the Gradient as SavedGradient
	parser.add_argument('--saveImage', default=True, type=bool) # True if want to save the Image of the Gradient
	parser.add_argument('--directoryFile', default="SavedGradient", type=str) # name of the directory where SavedGradient is saved
	parser.add_argument('--directoryImage', default="Gradient_output", type=str) # name of the output directory that will contain the image
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
	
	print('\n')

	# Name of the model files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i)+'_steps' for i in range(args.min_iter,
																	args.max_iter,
																	args.step_iter)]

	# Compute fitness over these directions :
	previous_theta = None # Stores theta
	newGradient = SavedGradient(directions=[], results=[], red_markers=[], green_markers=[],
								nbLines=args.line_height, pixelWidth=args.pixelWidth, pixelHeight=args.pixelHeight, maxValue=args.maxValue,
								dotText=args.dotText, dotWidth=args.dotWidth, xMargin=args.xMargin, yMargin=int(args.pixelHeight/2)) # Storing new SavedGradient
	for indice_file in range(len(filename_list)):

		# Change which model to load
		filename = filename_list[indice_file]

		# Load the model
		print("\nSTARTING : "+str(filename))
		model = SAC.load("{}/{}".format(args.directory, filename))
		
		# Get the new parameters
		theta0 = model.policy.parameters_to_vector()
		base_vect = theta0 if previous_theta is None else theta0 - previous_theta
		previous_theta = theta0
		print("Loaded parameters from file")

		# Evaluate the Model : mean, std
		print("Evaluating the model...")
		init_score, score_std = evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)
		print("Model initial fitness : ", (init_score, score_std))

		# Study the geometry around the model
		print("Starting study around the model...")
		#	Norm of the model
		length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
		# 		Direction taken by the model (normalized)
		d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist
		newGradient.directions.append(d)
		#		New parameters following the direction
		theta_plus, theta_minus = getPointsDirection(theta0, num_params, args.minalpha, args.maxalpha, args.stepalpha, d)

		# Evaluate using new parameters
		scores_plus, scores_minus = [], []
		with SlowBar("Evaluating along its direction", max=len(theta_plus)) as bar:
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

		# Adding the results
		last_params_marker = int(length_dist/args.stepalpha)
		#	Mark two consecutive positions on the line
		marker_actor = int((len(line)-1)/2)
		marker_last = max(marker_actor-last_params_marker, 0)
		#		A list of the markers, previous will be shown in red and current in green
		newGradient.red_markers.append(marker_last)
		newGradient.green_markers.append(marker_actor)
		# 	Putting it all together
		newGradient.results.append(line)
	
	try:
		# Assembling the image, saving it if asked
		newGradient.computeImage(saveImage=args.saveImage, filename=args.basename+'_gradient', directory=args.directoryImage)
	except Exception as e:
		newGradient.saveGradient(args.basename, args.directoryFile+'/temp')
		
	# Saving the SavedGradient if asked
	if args.saveFile is True: newGradient.saveGradient(args.basename, args.directoryFile)

	env.close()
