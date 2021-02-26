# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse
from progress.bar import Bar

import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from savedPlot import SavedPlot
from vector_util import *
from slowBar import SlowBar

# To test
# python3 Vignette.py --directory Ex_Sauvegarde/Saves --basename save --min_iter 1 --max_iter 10 --eval_maxiter 10 --plot3D True --show3D True
	
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
	#	3D plot parameters
	parser.add_argument('--x_diff', default=2., type=float)# the space between each point along the x-axis
	parser.add_argument('--y_diff', default=2., type=float)# the space between each point along the y-axis
	parser.add_argument('--line_width', default=1., type=float)# the width of each line
	parser.add_argument('--plot3D', default=False, type=bool)# true if an image of the plot needs to be saved
	parser.add_argument('--show3D', default=True, type=bool)# true if the plot needs to be shown
	parser.add_argument('--step3D', default=False, type=bool)# true if want to show the plot after each file (suspends execution)
	#		Save plot parameters
	parser.add_argument('--savePlot', default=True, type=bool)# true if want to save Plot for later use
	parser.add_argument('--saveFolder', default="Saved_plots", type=str)# name of the folder to save the plot in
	
	# File management
	parser.add_argument('--directory', default="TEST_5", type=str)# name of the directory containing the models to load
	parser.add_argument('--basename', default="model_sac_step_1_", type=str)# file prefix for the loaded model
	parser.add_argument('--min_iter', default=1, type=int)# iteration (file suffix) of the first model
	parser.add_argument('--max_iter', default=10, type=int)# iteration (file suffix) of the last model
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive models
	parser.add_argument('--base_output_filename', default="vignette_output", type=str)# name of the output file to create
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
	for indice_file in range(len(filename_list)):

		# Intitializing the 3D plot
		if args.plot3D or args.show3D is True:
			fig, ax = plt.figure(), plt.axes(projection="3d")
			

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

		# Iterating over all directions, -1 is the direction that was initially taken by the model
		newSave = SavedPlot(x_diff=args.x_diff, y_diff=args.y_diff, line_width=args.line_width) if args.savePlot is True else None # Creating a new save if asked
		for step in range(-1,len(D)):
			print("\nDirection ", step, "/", len(D))
			# New parameters following the direction
			theta_plus, theta_minus = getPointsDirection(theta0, num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
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
			if step == -1:	base_image.append(line)
			else:	image.append(line)
			#	Adding the line to the plot
			if args.plot3D or args.show3D or args.savePlot is True:
				x_line, y_line = np.linspace(-len(line)/2,len(line)/2,len(line)), np.ones(len(line))
				# Inverting y_line because Vignette reads from top to bottom
				height = -step if step != -1 else -len(D)-1
				ax.plot3D(args.x_diff * x_line, args.y_diff * height * y_line, line)
				
				# Adding to the save
				if args.savePlot is True:
					newSave.axes[step] = ax
					newSave.lines[step] = line

		# Assemble the image
		# 	Dark line separating the base and the directions
		separating_line = np.zeros(len(base_image[0]))
		last_params_marker = int(length_dist/args.stepalpha)
		marker_pixel = int((len(base_image[0])-1)/2-last_params_marker)
		separating_line[marker_pixel] = v_max_fit
		#		Concatenation, repeating each line 10 times for visibility
		final_image = np.concatenate((image, [separating_line], base_image), axis=0)
		final_image = np.repeat(final_image,10,axis=0)#repeating each line 10 times to be visible
		final_image = np.repeat(final_image,10,axis=1)#repeating each line 10 times to be visible
		#			Saving the image
		plt.imsave(args.base_output_filename+"_"+str(filename)+".png",final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

		# Saving an image of the 3D plot if asked
		if args.plot3D is True: plt.savefig("3D_"+args.base_output_filename+"_"+str(filename)+".png")
		# 	Showing the 3D plot if asked (suspends execution)
		if args.step3D is True: plt.show()
		#		Saving the plot in a file for later use if asked
		if args.savePlot is True: newSave.saveInFile("Plot_"+args.base_output_filename+"_"+str(filename), args.saveFolder)
		
	# Showing all the plots if asked
	if args.show3D is True: plt.show()


	env.close()
