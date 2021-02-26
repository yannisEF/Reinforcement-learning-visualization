# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse

from progress.bar import Bar
from PIL import Image
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

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
	previous_theta = None # Remembers theta
	directions = [] # Remembers the directions taken
	red_markers, yellow_markers = [], [] # Remembers the path taken by the model
										 # yellow is an image of the current parameters
										 # red is and image of the previous parameters
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
		theta_plus_scores, theta_minus_scores = [], []
		image, base_image = [], []

		#	Norm of the model
		length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
		# 		Direction taken by the model (normalized)
		d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist
		directions.append(5*d)
		#		New parameters following the direction
		theta_plus, theta_minus = getPointsDirection(theta0, num_params, args.minalpha, args.maxalpha, args.stepalpha, d)

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

		# Adding the results
		last_params_marker = int(length_dist/args.stepalpha)
		#	Mark two consecutive positions on the line
		marker_actor = int((len(line)-1)/2)
		marker_last = max(marker_actor-last_params_marker, 0)
		#		Dark separation between each line, contains the models' markers
		separating_line = np.array([v_min_fit]*len(line))
		separating_line[marker_last] = (v_max_fit+v_min_fit)/2
		separating_line[marker_actor] = v_max_fit
		#		A list of the markers, yellow is current for each line, red is previous
		red_markers.append(marker_last)
		yellow_markers.append(marker_actor)
		# 	Putting it all together
		results.append(separating_line)
		results.append(line)
	
	# Assembling the image
	final_image = np.repeat(results,args.pixelHeight,axis=0)
	final_image = np.repeat(final_image,args.pixelWidth,axis=1)
	plt.imsave(args.image_filename,final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

	# Adding a side-image showing the dot product
	im = Image.open(args.image_filename)
	width, height = im.size
	output = Image.new("RGB",(width+170, height))
	for y in range(height):
		for x in range(width):
			output.putpixel((x,y),im.getpixel((x,y)))
	for nb_m in range(len(red_markers)):# red markers
		for y in range(10):
			for x in range(20):
				output.putpixel((x+20*red_markers[nb_m],y+nb_m*40),(255,0,0))

	for i in range(0,len(filename_list)-1):# dot product values
		scalar_product = np.dot(directions[i], directions[i+1])
		print("dot prod : "+str(scalar_product))
		color = (0,255,0)
		if(scalar_product < -0.2):
			color = (255,0,0)
		else :
			if (scalar_product >-0.2 and scalar_product < 0.2):
				color = (255,140,0)
		for j in range(min(int(10*abs(scalar_product))+10,150)):
			for k in range(1,20):
				output.putpixel((width+j+10,i*40+30+k),color)

	env.close()
	
