from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from ES import sepCEM, Control
from models import RLNN
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from util import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


save_average_mu_in_csv = True #enregistrer les différents mu average de l'actor dans un csv pendant l'apprentissage
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor #?

def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
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
    speeds = []#approximately scores
    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False
        speed_episode = []
        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            speed_episode.append(reward)
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
        speeds.append(speed_episode)
        scores.append(score)
    return np.mean(scores), steps, speeds


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

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

class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = torch.nn.functional.relu(self.l1(torch.cat([x, u], 1)))
            x1 = torch.nn.functional.relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = torch.nn.functional.relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = torch.nn.functional.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = torch.nn.functional.relu(self.l4(torch.cat([x, u], 1)))
            x2 = torch.nn.functional.relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = torch.nn.functional.relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = torch.nn.functional.relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='Swimmer-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_false')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--mem_size', default=10000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=1000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()

    print("creating environment")
    # environment
    env = gym.make(args.env)

    print("starting...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    #Theses informations are about reading the files of the actors :
    td3_steps = []
    td3_maxQ = []
    td3_maxP = []
    td3_meanQ = []
    td3_meanP = []

    NB_RUNS = 10
    directory = "TEST_5"
    filename_base_1 = "_td3_"
    filename_base_2 = "_step_1"
    TD3_STEP = 1000
    TD3_END = 200000
    start_steps=20000

    #loading TD3
    print("Loading TD3 results...")
    for i in range(1,NB_RUNS+1):
        td3_steps = []
        td3_maxQ = []
        td3_maxP = []
        td3_meanQ = []
        td3_meanP = []
        fits = []
        for j in range(TD3_STEP,TD3_END+TD3_STEP,TD3_STEP):
            filename = filename_base_1+str(i)+filename_base_2+"_"+str(j)
            actor = Actor(state_dim, action_dim, max_action,layer_norm=args.layer_norm)
            actor.load_model(directory,"actor"+filename)

            fit, _, speeds = evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False)
            
            paramsA = np.array(actor.get_params())
            fits.append(fit)
            td3_steps.append(j)
        plt.figure()
        random = plt.axvline(x=start_steps, color = "g", linestyle="dashed")
        fitness = plt.plot(td3_steps, fits, color = "g")
        plt.grid()
        plt.legend([fitness, random], ["actor fitness", "random policy end"])
        plt.show()

    print("Ending...")
    env.close()
