# coding: utf-8

import numpy as np
from copy import deepcopy
from models import RLNN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import gym.spaces
from memory import Memory
from util import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (atanh(action / self.action_range)
               if action is not None
               else dist.rsample())
        act_entropy = dist.entropy()

        # the suggested way to confine your actions within a valid range
        # is not clamping, but remapping the distribution
        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range

        # the distribution remapping process used in the original essay.
        act_log_prob -= t.log(self.action_range *
                              (1 - act_tanh.pow(2)) +
                              1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        # If your distribution is different from "Normal" then you may either:
        # 1. deduce the remapping function for your distribution and clamping
        #    function such as tanh
        # 2. clamp you action, but please take care:
        #    1. do not clamp actions before calculating their log probability,
        #       because the log probability of clamped actions might will be
        #       extremely small, and will cause nan
        #    2. do not clamp actions after sampling and before storing them in
        #       the replay buffer, because during update, log probability will
        #       be re-evaluated they might also be extremely small, and network
        #       will "nan". (might happen in PPO, not in SAC because there is
        #       no re-evaluation)
        # Only clamp actions sent to the environment, this is equivalent to
        # change the action reward distribution, will not cause "nan", but
        # this makes your training environment further differ from you real
        # environment.
        return act, act_log_prob, act_entropy

    def save_model(self, output, net_name):
        t.save(self.state_dict(),'{}/{}.pkl'.format(output, net_name))
    
    def load_model(self, filename, net_name):
    """
    Loads the model
    """
    	if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False, maxiter=1000): #init evaluation
    """
    Computes the score of an actor on a given number of runs, on a giver number of steps (1000 for full episode evaluation,
    500 for 1/2 épisodes evaluations...)
    fills the replay buffer if given (not None)
    """

    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor(state).cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):
        score = 0
        obs = deepcopy(env.reset())
        done = False
        iter_nb = 0
        while not done and iter_nb<maxiter:
            iter_nb+=1
            # get next action and act
            action = policy(obs)
            n_obs, reward, done, info = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()
        scores.append(score)

    return np.mean(scores), steps


def getPointsChoice(init_params,num_params, minalpha, maxaplha, stepalpha, prob):
	"""
	# Params :

	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	prob : the probability to choose each parameter dimension (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by random choice of proba 'prob' on param dimensions.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives a good but very noisy visualisation and not easy to interpret.
	"""
	#init_params = np.copy(base_params)
	d = np.random.choice([1, 0], size=(num_params,), p=[prob, 1-prob]) #select random dimensions with proba 
	print("d: "+str(d))
	print("proportion: "+str(np.count_nonzero(d==1))+"/"+str(num_params))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniform(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :

	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [0,1).
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives the best visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(0, 1, num_params) #select uniformly dimensions [0,1)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsDirection(init_params,num_params, minalpha, maxaplha,stepalpha, d):
	"""
	# Params :

	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	d : pre-choosend direction
	
	# Function:

	Returns parameters around base_params on direction given in parameters.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives an output that is comparable with other results if directions are the same.
	"""
	#init_params = np.copy(base_params)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniformCentered(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :

	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [-1,1).
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha. 
	This method gives bad visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(-1, 1, num_params) #select uniformly dimensions in [-1,1)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getDirectionsMuller(nb_directions,num_params):
    """
    # Params :

    nb_directions : number of directions to generate randomly in unit ball
    num_params : dimensions of the vectors to generate (int value, only 1D vectors)
	
    # Function:

    Returns a list of vectors generated in the uni ball of 'num_params' dimensions, using Muller
    """
    D = []
    for _ in range(nb_directions):
        u = np.random.normal(0,1,num_params)
        norm = np.sum(u**2)**(0.5)
        r = np.random.random()**(1.0/num_params)
        x = r*u/norm
        print("vect muller:"+str(x))
        print("euclidian dist:"+str(euclidienne(x, np.zeros(len(x)))))
        D.append(x)
    return D

def euclidienne(x,y):
    """
    # Params :

    x,y : vectors of the same size
	
    # Function:

    Returns a simple euclidian distance between x and y.
    """
    return np.linalg.norm(np.array(x)-np.array(y))

def order_all_by_proximity(vectors):
    """
    # Params :

    vectors : a list of vectors
	
    # Function:

    Returns the list of vectors ordered by inserting the vectors between their nearest neighbors
    """
    ordered = []
    for vect in vectors :
        if(len(ordered)==0):
            ordered.append(vect)
        else:
            ind = compute_best_insert_place(vect, ordered)
            ordered.insert(ind,vect)
    return ordered

def compute_best_insert_place(vect, ordered_vectors):
    """
    # Params :

    ordered_vectors : a list of vectors ordered by inserting the vectors between their nearest neighbors
    vect : a vector to insert at the best place in the ordered list of vectors
	
    # Function:

    Returns the index where 'vect' should be inserted to be between the two nearest neighbors using euclidien distance
    """
    # Compute the index where the vector will be at the best place :
    value_dist = euclidienne(vect, ordered_vectors[0])
    dist_place = [value_dist]
    for ind in range(len(ordered_vectors)-1):
        value_dist = np.mean([euclidienne(vect, ordered_vectors[ind]),euclidienne(vect, ordered_vectors[ind+1])])
        dist_place.append(value_dist)
    value_dist = euclidienne(vect, ordered_vectors[len(ordered_vectors)-1])
    dist_place.append(value_dist)
    ind = np.argmin(dist_place)
    return ind
	
if __name__ == "__main__":

	print("Parsing arguments")

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Swimmer-v2', type=str)
	parser.add_argument('--tau', default=1, type=float) #for initializing the actor, not used really
	parser.add_argument('--layer_norm', dest='layer_norm', action='store_true') #for initialising the actor
	parser.add_argument('--nb_lines', default=60, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
	parser.add_argument('--discount', default=0.99, type=float) #for initializing the actor, not used really
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : large 100, around actor 10
	parser.add_argument('--stepalpha', default=0.25, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
	parser.add_argument('--eval_maxiter', default=1000, type=float)# number of steps for the evaluation. Depends on environment.
	parser.add_argument('--min_colormap', default=-10, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--max_colormap', default=360, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--proba', default=0.1, type=float)# proba of choosing an element of the actor parameters for the direction, if using the choice method.
	parser.add_argument('--basename', default="ok", type=str)# base (files prefix) name of the actor pkl files to load
	parser.add_argument('--min_iter', default=1000, type=int)# iteration (file suffix) of the first actor pkl files to load
	parser.add_argument('--max_iter', default=200000, type=int)# iteration (file suffix) of the last actor pkl files to load
	parser.add_argument('--step_iter', default=1000, type=int)# iteration step between two consecutive actor pkl files to load
	parser.add_argument('--base_output_filename', default="vignette_output", type=str)# name of the output file to create
	parser.add_argument('--epsilon', default=10, type=float) #for initialising the actor, not used really
	parser.add_argument('--filename', default="TEST_5", type=str)# name of the directory containing the actors pkl files to load
	parser.add_argument('--actor_lr', default=0.001, type=float) #for initializing the actor, not used really
	parser.add_argument('--critic_lr', default=0.001, type=float) #for initializing the actor, not used really
	args = parser.parse_args()

	# Creating environment and initialising actor and parameters
	print("Creating environment")
	env = gym.make(args.env) #on Swimmer, 1/3 eval is enouth to have a good estimation of the reward of an actor
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])
	actor = Actor(state_dim, action_dim, max_action, args)
	theta0 = actor.get_params()
	num_params = len(theta0)
	v_min_fit = args.min_colormap
	v_max_fit = args.max_colormap
	print("VMAX :"+str(v_max_fit))

	# Choosing directions
	#D = np.random.rand(args.nb_lines,num_params)
	D = getDirectionsMuller(args.nb_lines,num_params)

	# Ordering the directions :
	D = order_all_by_proximity(D)

	# Name of the actor files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i) for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate actor file list to load
	# Compute fitness over these directions :
	last_actor_params = [] #save the last parameters, to compute direction followed from the previous actor
	for indice_file in range(len(filename_list)):
		filename = filename_list[indice_file]
		# Loading actor params
		print("STARTING : "+str(filename))
		actor = Actor(state_dim, action_dim, max_action, args) #removed to study start point only
		actor.load_model(args.filename, filename)
		theta0 = actor.get_params()
		if(len(last_actor_params)>0):
			previous = last_actor_params
			base_vect = theta0 - previous #compute direction followed from the previous actor
			last_actor_params = theta0 #update last_actor_params
		else:
			base_vect = theta0 #if first actor (no previous), considering null vector is the previous
			last_actor_params = theta0 #update last_actor_params
		print("params : "+str(theta0))
		# evaluate the actor
		init_score, _ = evaluate(actor, env, maxiter=args.eval_maxiter)
		epsilon = args.epsilon
		print("Actor initial fitness : "+str(init_score))
		# Running geometry study around the actor
		print("Starting study aroud...")
		theta_plus_scores = []
		theta_minus_scores = []
		image = []
		base_image = []
		
		### Direction followed from precedent actor :
		length_dist = euclidienne(base_vect, np.zeros(len(base_vect)))
		d= base_vect / length_dist #reduce to unit vector
		theta_plus, theta_minus = getPointsDirection(theta0,num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
		temp_scores_theta_plus = []
		temp_scores_theta_minus = []
		for param_i in range(len(theta_plus)):
			# we evaluate the actor (theta_plus) :
			actor.set_params(theta_plus[param_i])
			score_plus,_ = evaluate(actor, env,maxiter=args.eval_maxiter)
			temp_scores_theta_plus.append(score_plus)
			# we evaluate the actor (theta_minus) :
			actor.set_params(theta_minus[param_i])
			score_minus,_ = evaluate(actor, env,maxiter=args.eval_maxiter)
			temp_scores_theta_minus.append(score_minus)
		#we invert scores on theta_minus list to display symetricaly the image with init params at center,
		# theta_minus side on the left and to theta_plus side on the right
		buff_inverted = np.flip(temp_scores_theta_minus)
		plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
		base_image.append(plot_pixels)#adding these results as a line in the output image
		#saving the score values
		theta_plus_scores.append(temp_scores_theta_plus)
		theta_minus_scores.append(temp_scores_theta_minus)

		### Directions chosen
		for step in range(len(D)) :
			print("step "+str(step))
			#computing actor parameters
			d = D[step]
			theta_plus, theta_minus = getPointsDirection(theta0,num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
			temp_scores_theta_plus = []
			temp_scores_theta_minus = []
			for param_i in range(len(theta_plus)):
				# we evaluate the actor (theta_plus) :
				actor.set_params(theta_plus[param_i])
				score_plus,_ = evaluate(actor, env,maxiter=args.eval_maxiter)
				#print("score plus : "+str(score_plus))
				temp_scores_theta_plus.append(score_plus)
				# we evaluate the actor (theta_minus) :
				actor.set_params(theta_minus[param_i])
				score_minus,_ = evaluate(actor, env,maxiter=args.eval_maxiter)
				temp_scores_theta_minus.append(score_minus)
			#we invert scores on theta_minus list to display symetricaly the image with init params at center,
			# theta_minus side on the left and to theta_plus side on the right
			buff_inverted = np.flip(temp_scores_theta_minus)
			plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
			image.append(plot_pixels)#adding these results as a line in the output image
			#saving the score values
			theta_plus_scores.append(temp_scores_theta_plus)
			theta_minus_scores.append(temp_scores_theta_minus)
		#assemble picture from different parts (choosen directions, dark line for separating, and followed direction)
		separating_line = np.zeros(len(base_image[0]))
		last_params_marker = int(length_dist/args.stepalpha)
		marker_pixel = int((len(base_image[0])-1)/2-last_params_marker)
		separating_line[marker_pixel] = v_max_fit
		final_image = np.concatenate((image, [separating_line], base_image), axis=0)
		#showing final result
		final_image = np.repeat(final_image,10,axis=0)#repeating each line 10 times to be visible
		final_image = np.repeat(final_image,10,axis=1)#repeating each line 10 times to be visible
		plt.imsave(args.base_output_filename+"_"+str(filename)+".png",final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

	env.close()



	