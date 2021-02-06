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
    Computes the score of an actor on a given number of runs, on a giver number of steps (1000 for full episode evaluation,
    500 for 1/2 épisodes evaluations...)
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

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Swimmer-v2', type=str)
	parser.add_argument('--tau', default=0.005, type=float) #for initialising the actor, not used really
	parser.add_argument('--layer_norm', dest='layer_norm', action='store_true') #for initialising the actor
	parser.add_argument('--max_steps', default=50, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
	parser.add_argument('--discount', default=0.99, type=float) #for initialising the actor, not used really
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=10, type=float)# end value for alpha, good value : precise 120, fast 100, ultrafast 100
	parser.add_argument('--stepalpha', default=0.5, type=float)# step for alpha in the loop, good value : precise 1, fast 2, ultrafast 3
	parser.add_argument('--eval_maxiter', default=1000, type=float)# number of steps for the evaluation. Depends on environment episode length. On Swimmer, full eval : 1000, 1/2 eval : 500, 1/3 eval : 300 ... (faster but more aproximated)
	parser.add_argument('--proba', default=0.1, type=float)# proba of choosing an element of the actor parameters for the direction, if using the choice method.
	parser.add_argument('--epsilon', default=10, type=float) #for initialising the actor, not used really
	parser.add_argument('--filename', default="TEST_4", type=str)# name of the directory containing the actors pkl files to load
	parser.add_argument('--actor_lr', default=0.001, type=float) #for initialising the actor, not used really
	parser.add_argument('--critic_lr', default=0.001, type=float) #for initialising the actor, not used really
	args = parser.parse_args() #actors_start_sans_modif

	# Creating environment and initialising actor and parameters
	print("Creating environment")
	env = gym.make(args.env) #on Swimmer, 1/3 eval is enouth to have a good estimation of the reward of an actor
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])
	actor = Actor(state_dim, action_dim, max_action, args)
	theta0 = actor.get_params()
	num_params = len(theta0)
	v_min_fit = -10 # min fitness
	v_max_fit = 360/(1000/args.eval_maxiter) #adapting max fitness on number of evaluation steps (aproximatively)
	print("VMAX :"+str(v_max_fit))

	# Name of the actor files to analyse consecutively with the same set of directions: 
	#filename_list = ["actor_td3_buffer_NIGHT_1_step_1_91000","actor_td3_buffer_NIGHT_1_step_1_96000","actor_td3_buffer_NIGHT_1_step_1_101000","actor_td3_buffer_NIGHT_1_step_1_106000"]
	filename_list = [["actor_td3_"+str(j)+"_step_1_"+str(i) for i in range(1000,201000,1000)] for j in range(1,5)]
	#filename_list.append(["actor_td3_2_step_1_"+str(i) for i in range(1000,201000,1000)])#start = 5000 /!\ modified, don't forget to put it back
	image_filename = "gradient_td3_2_gamma_1.png" #output picture  
	# Compute fitness over these directions :
	last_actor_params = [] #save the last parameters, to compute direction followed from the precedent actor
	result = []
	directions = []
	dot_values = []
	yellow_markers = []
	red_markers = []
	distances = []
	distance_run = []
	for run in range(len(filename_list)):
		last_actor_params = []
		for indice_file in range(len(filename_list[run])):
			filename = filename_list[run][indice_file]
			# Loading actor params
			print("FILE : "+str(filename))
			actor = Actor(state_dim, action_dim, max_action, args)
			actor.load_model(args.filename, filename)
			theta0 = actor.get_params()
			if(len(last_actor_params)>0):
				previous = last_actor_params
				base_vect = theta0 - previous #compute direction followed from the precedent actor
				last_actor_params = theta0 #update last_actor_params
				### Distance from precedent actor :
				length_dist = euclidienne(base_vect, np.zeros(len(base_vect)))
				distance_run.append(length_dist)
			else:
				base_vect = theta0 #if first actor (no precedent), considering null vector is the precedent
				last_actor_params = theta0 #update last_actor_params
			#directions.append(base_vect/np.max(abs(base_vect)))#save unity vector of estimated gradient direction
			print("params : "+str(theta0))
		

			#showing final result
		distances.append(distance_run)
	env.close()
	means = [np.mean([distances[i][k] for i in range(len(filename_list))]) for k in range(len(filename_list[0]))]
	plt.figure()
	dist, = plt.plot(list(range(1000,201000,1000)), means, label="Euclidean distance between consecutive actors on Swimmer")
	dist_mean = plt.hlines(np.mean(means),1000,200000, color='r', lineStyle='dotted', label= "Average distance")
	plt.legend([dist, dist_mean],["Euclidean distance between consecutive actors on Swimmer","Average distance"])
	plt.show()



	
