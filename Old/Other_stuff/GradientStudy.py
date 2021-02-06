import numpy as np
from copy import deepcopy
from models import RLNN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import gym.spaces
from memory import Memory
from util import *
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False, maxiter=1000): #init evaluation
    """
    Computes the score of an actor on a given number of runs, on a giver number of steps
    fills the replay buffer if given (not None)
    """

    if not random:
        def policy(state): # Prise de décision de l'action à effectuer => ça nous intéresse
            state = FloatTensor(state.reshape(-1))
            action = actor(state).cpu().data.numpy().flatten() # ???

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
		#print(alpha)
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


if __name__ == "__main__":

	print("Parsing arguments")

	# Parameters are tuned for TD3 actors taken every 1000 steps, on Swimmer benchmark.
	# About loading the actors files :
	#   The actor filenames we use have a common prefix (actor_td3_1_step_1 for example) followed by the number of iterations.
	#   example : actor_td3_1_step_1_1000, actor_td3_1_step_1_25000, or myactor_200000. 
	#   Check parameters 'basename', 'min_iter, 'max_iter' and step_iter.
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Swimmer-v2', type=str)
	parser.add_argument('--tau', default=0.005, type=float) #for initializing the actor, not used really
	parser.add_argument('--layer_norm', dest='layer_norm', action='store_true') #for initializing the actor
	parser.add_argument('--discount', default=0.99, type=float) #for initializing the actor, not used really
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : precise 120, fast 100, ultrafast 100
	parser.add_argument('--stepalpha', default=0.5, type=float)# step for alpha in the loop, good value : precise 1, fast 2, ultrafast 3
	parser.add_argument('--eval_maxiter', default=1000, type=float)# number of steps for the evaluation.
	parser.add_argument('--min_colormap', default=-10, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--max_colormap', default=360, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--epsilon', default=10, type=float) #for initializing the actor, not used really
	parser.add_argument('--filename', default="TEST_5", type=str)# name of the directory containing the actors pkl files to load
	parser.add_argument('--basename', default="actor_td3_2_step_1_", type=str)# base (files prefix) name of the actor pkl files to load
	parser.add_argument('--min_iter', default=1000, type=int)# iteration (file suffix) of the first actor pkl files to load
	parser.add_argument('--max_iter', default=201000, type=int)# iteration (file suffix) of the last actor pkl files to load
	parser.add_argument('--step_iter', default=1000, type=int)# iteration step between two consecutive actor pkl files to load
	parser.add_argument('--output_filename', default="gradient_output.png", type=str)# name of the output file to create
	parser.add_argument('--actor_lr', default=0.001, type=float) #for initializing the actor, not used really
	parser.add_argument('--critic_lr', default=0.001, type=float) #for initialzing the actor, not used really
	args = parser.parse_args()

	# Creating environment and initialising actor and parameters
	print("Creating environment")
	env = gym.make(args.env)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])
	actor = Actor(state_dim, action_dim, max_action, args)
	theta0 = actor.get_params()
	num_params = len(theta0)
	v_min_fit = args.min_colormap
	v_max_fit = args.max_colormap
	print("VMAX :"+str(v_max_fit))

	# Name of the actor files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i) for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate actor file list to load
	image_filename = args.output_filename #output picture  
	# Compute fitness over these directions :
	last_actor_params = [] #s ave the last parameters, to compute direction followed from the precedent actor
	result = []
	directions = []
	dot_values = []
	yellow_markers = []
	red_markers = []
	for indice_file in range(len(filename_list)):
		filename = filename_list[indice_file]
		# Loading actor params
		print("FILE : "+str(filename))
		actor = Actor(state_dim, action_dim, max_action, args)
		actor.load_model(args.filename, filename)
		theta0 = actor.get_params()
		if(len(last_actor_params)>0):
			previous = last_actor_params
			base_vect = theta0 - previous # compute direction followed from the precedent actor
			last_actor_params = theta0 # update last_actor_params
		else:
			base_vect = theta0 # if first actor (no precedent), considering null vector is the precedent
			last_actor_params = theta0 # update last_actor_params

		print("params : "+str(theta0))
		# evaluate the actor
		init_score, _ = evaluate(actor, env, maxiter=args.eval_maxiter)
		epsilon = args.epsilon
		print("Actor initial fitness : "+str(init_score))
		# Running geometry study around the actor
		print("Computing fitness on this direction...")
		theta_plus_scores = []
		theta_minus_scores = []
		base_image = []
		
		### Direction followed from precedent actor :
		length_dist = euclidienne(base_vect, np.zeros(len(base_vect)))
		print("length_dist : "+str(length_dist))
		if length_dist != 0 :
			d= base_vect / abs(length_dist) # reduce to unit vector
		else:
			d = np.zeros(len(base_vect))
		directions.append(d*5)# save unity vector of estimated gradient direction
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
		# we invert scores on theta_minus list to display symetricaly the image with init params at center,
		# theta_minus side on the left and to theta_plus side on the right
		buff_inverted = np.flip(temp_scores_theta_minus)
		plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
		# saving the score values
		theta_plus_scores.append(temp_scores_theta_plus)
		theta_minus_scores.append(temp_scores_theta_minus)

		
		# assemble picture from different parts (choosen directions, dark line for separating, and followed direction)
		mean_value = (v_max_fit-v_min_fit)/2+v_min_fit
		separating_line = np.array([v_min_fit]*len(plot_pixels))
		last_params_marker = int(length_dist/args.stepalpha)
		if last_params_marker < 0 :
			marker_last = min( int((len(plot_pixels)-1)/2+last_params_marker) , len(plot_pixels)-1)
		else:
			marker_last = max( int((len(plot_pixels)-1)/2-last_params_marker), 0)
		marker_actor = int((len(plot_pixels)-1)/2)
		yellow_markers.append(marker_actor)
		red_markers.append(marker_last)
		separating_line[marker_last] = mean_value # previous actor in blue (original version, modified by red markers below)
		separating_line[marker_actor] = v_max_fit # current actor in yellow
		result.append(separating_line)# separating line
		result.append(plot_pixels) # adding multiple time the same entry, to better see
		result.append(plot_pixels)
		result.append(plot_pixels)
	
	# preparing final result
	final_image = np.repeat(result,10,axis=0)# repeating each 10 times to be visible
	final_image = np.repeat(final_image,20,axis=1)# repeating each 20 times to be visible

	plt.imsave(image_filename,final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

	env.close()

	# adding dot product & markers visual infos : 
	im = Image.open(image_filename)
	width, heigh = im.size
	output = Image.new("RGB",(width+170, heigh))
	for y in range(heigh):
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

	# saving result
	output.save(image_filename,"PNG")



	
