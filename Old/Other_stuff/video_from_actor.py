    ###########################################################################################################################
    ######   Ce code permet de sauvegarder la vidéo d'un run sur Swimmer d'un acteur chargé depuis un fichier pkl    ##########
    ###########################################################################################################################

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


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

    #####################################################
    ## Classe de l'acteur (pour pas avoir à importer): ##
    #####################################################

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


if __name__ == "__main__":

    ###########################################
    ## Création des arguments pour l'acteur: ##
    ###########################################

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
    parser.add_argument('--max_steps', default=500000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()

    #########################################
    ##### Vidéo de l'acteur sur Swimmer: ####
    #########################################
    #
    # Note : on cherche ici à faire fonctionner le Swimmer en le branchant à l'actor qui a été entrainé.
    # On commence par réinitialiser un environnement Swimmer totalement neutre, puis on appelle
    # l'acteur pour choisir l'action à faire, et on sauvegarde le résultat en vidéo mp4.
    
    #paramètres :
    output_video_dir = './video_actor' # /!\ Nom du dossier où enregistrer la vidéo
    actor_directory = "actors" # /!\ dossier où se trouve l'acteur
    actor_filename = "TD3_310" # /!\ nom du fichier de l'acteur (sans l'extension)
    
    #création d'un environnement Swimmer :
    print("Creating environment")
    env = gym.make('Swimmer-v2')

    #ajout d'un moniteur pour l'enregistrement vidéo sur l'environnement :
    env = gym.wrappers.Monitor(env, output_video_dir,force=True)

    #itialisation de l'environnement :
    env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    action = [0. , 0.]
    state, reward, done, _ = env.step(action)#on récupère l'état en faisant une action "nulle", petite triche.
    
    #chargement de l'acteur :
    print("Loading actor")
    actor = Actor(state_dim, action_dim, max_action, args)
    actor.load_model(actor_directory,actor_filename)

    #itérations sur les actions choisies par l'acteur entrainté :
    print("Running...")
    while not done:
        #conversion de l'état dans le bon format (FloatTensor) pour la méthode forward de Actor:
        stateTorch = FloatTensor(np.array(state).reshape(-1))
        #choix de l'action par l'acteur entrainté :
        action = actor.forward(stateTorch).cpu().data.numpy().flatten()
        #on effectue cette action et on récupère le nouvel état :
        state, reward, done, _ = env.step(action)
    print("Video saved in : "+str(output_video_dir))

    #fermeture de l'environnement:
    env.close()
