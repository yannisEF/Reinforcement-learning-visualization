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


# To test
# python3 GradientStudy.py --directory Ex_Sauvegarde/Saves --basename save --min_iter 1 --max_iter 5 --eval_maxiter 1

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

	# Tools parameters
	parser.add_argument('--nb_lines', default=60, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : large 100, around model 10
	parser.add_argument('--stepalpha', default=0.25, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
	parser.add_argument('--eval_maxiter', default=1000, type=float)# number of steps for the evaluation. Depends on environment.
	parser.add_argument('--min_colormap', default=-10, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--max_colormap', default=360, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--pixelWidth', default=20, type=int)# width of each pixel
	parser.add_argument('--pixelHeight', default=10, type=int)# height of each pixel
	#	Dot product parameters
	parser.add_argument('--dotWidth', default=150, type=int)# max width of the dot product (added on the side)
	parser.add_argument('--dotText', default=True, type=str)# true if want to show value of the dot product

	# File management
	parser.add_argument('--directory', default="TEST_5", type=str)# name of the directory containing the models to load
	parser.add_argument('--basename', default="model_sac_step_1_", type=str)# file prefix for the loaded model
	parser.add_argument('--min_iter', default=1, type=int)# iteration (file suffix) of the first model
	parser.add_argument('--max_iter', default=10, type=int)# iteration (file suffix) of the last model
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive models
	parser.add_argument('--image_filename', default="gradient_output", type=str)# name of the output file to create
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
	
	# Plotting parameters
	v_min_fit = args.min_colormap
	v_max_fit = args.max_colormap

	# Choosing directions to follow
	D = getDirectionsMuller(args.nb_lines,num_params)
	# 	Ordering the directions :
	D = order_all_by_proximity(D)

	# Name of the model files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i) for i in range(args.min_iter,
														args.max_iter+args.step_iter,
														args.step_iter)]

	# Compute fitness over these directions :
	previous_theta = None # Stores theta
	directions = [] # Stores the directions taken
	red_markers, green_markers = [], [] # Stores the previous and current models' positions at each step
	results = [] # Stores each line
	for indice_file in range(len(filename_list)):

		# Change which model to load
		filename = filename_list[indice_file]

		# Load the model
		print("\nSTARTING : "+str(filename))
		model.load("{}/{}".format(args.directory, filename))
		
		# Get the new parameters
		theta0 = model.policy.parameters_to_vector()
		base_vect = theta0 if previous_theta is None else theta0 - previous_theta
		previous_theta = theta0
		print("Loaded parameters from file")

		# Evaluate the Model : mean, std
		init_score = evaluate_policy(model, env, n_eval_episodes=args.eval_maxiter, warn=False)[0]
		print("Model initial fitness : "+str(init_score))

		# Study the geometry around the model
		print("Starting study around the model...")
		#	Norm of the model
		length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
		# 		Direction taken by the model (normalized)
		d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist
		directions.append(d)
		#		New parameters following the direction
		theta_plus, theta_minus = getPointsDirection(theta0, num_params, args.minalpha, args.maxalpha, args.stepalpha, d)

		# Evaluate using new parameters
		scores_plus, scores_minus = [], []
		with SlowBar("Evaluating along the model's direction", max=len(theta_plus)) as bar:
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
		red_markers.append(marker_last)
		green_markers.append(marker_actor)
		# 	Putting it all together
		results.append(line)
	
	# Assembling the image
	nbLines = 3 # The height in number of pixels for each result
	width, height = args.pixelWidth * len(line), args.pixelHeight * len(results) * (nbLines+1)
	newIm = Image.new("RGB",(width+args.dotWidth, height))
	newDraw = ImageDraw.Draw(newIm)
	
	#	Putting the results and markers
	color1, color2 = colorTest.color1, colorTest.color2
	for l in range(len(results)):
		#	Separating lines containing the model's markers
		x0, y0 = red_markers[l] * args.pixelWidth, l * (nbLines+1) * args.pixelHeight
		x1, y1 = x0 + args.pixelWidth, y0 + args.pixelHeight
		newDraw.rectangle([x0, y0, x1, y1], fill=(255,0,0))

		x0 = green_markers[l] * args.pixelWidth
		x1 = x0 + args.pixelWidth
		newDraw.rectangle([x0, y0, x1, y1], fill=(0,255,0))

		# 	Drawing the results
		y0 += args.pixelHeight
		y1 = y0 + nbLines * args.pixelHeight
		for c in range(len(results[l])):
			x0 = c * args.pixelWidth
			x1 = x0 + args.pixelWidth
			color = valueToRGB(results[l][c], color1, color2, pureNorm=v_max_fit)
			newDraw.rectangle([x0, y0, x1, y1], fill=color)
		
		#	Processing the dot product,
		if l < len(results)-1:
			dot_product = np.dot(directions[l], directions[l+1])
			color = valueToRGB(dot_product, (255,0,0), (0,255,0), pureNorm=1)

			# Putting in on the side with a small margin
			xMargin, yMargin = 10, args.pixelHeight
			x0, y0Dot = xMargin + width, y1 - yMargin
			x1, y1Dot = x0 + min(abs(dot_product), args.dotWidth-xMargin), y1 + args.pixelHeight + yMargin
			newDraw.rectangle([x0, y0Dot, x1, y1Dot], fill=color)

			# Showing the value of the dot product if asked
			if args.dotText is True: newDraw.text((x0,y1), "{:.2f}".format(dot_product), fill=invertColor(color))

	newIm.save(args.image_filename+'.png', format='png')
	env.close()
	
