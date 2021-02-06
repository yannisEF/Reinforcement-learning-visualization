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

    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
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


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 40)
        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(40)
            self.n2 = nn.LayerNorm(30)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 40)
        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(40)
            self.n2 = nn.LayerNorm(30)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 40)
        self.l5 = nn.Linear(40, 30)
        self.l6 = nn.Linear(30, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(40)
            self.n5 = nn.LayerNorm(30)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


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

    print("creating environment")
    # environment
    env = gym.make(args.env)

    print("starting...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    global_data = []#contains all points 
    init_data = []#contains only starting points
    cem_data = []#contains only cem points
    td3_data = []#contains only td3 points

    fitness_start_point = []#contains only starting fitness
    fitness_cem = []#contains only cem fitness
    fitness_td3 = []#contains only td3 fitness

    #Theses informations are about reading the files of the actors :
    CEM_STEP = 10000 # step between two actors of the same run of CEM
    TD3_STEP = 5000 # step between two actors of the same run of TD3
    TD3_END = 150000 # last iteration number of TD3
    CEM_END = 200000 # last iteration number of CEM
    NB_RUNS = 4 #number of runs of each algorithm. Here, small for short test of compilation, but usualy 52.

    #loading Start Points
    print("Loading Starting Points...")
    for i in range(NB_RUNS):
        print("StartPoint : "+str(i))
        model = Actor(state_dim, action_dim, max_action, args)
        model.load_model("cem_explo_2","actor_cem_"+str(i)+"_0")
        params = model.get_params()
        fit = evaluate(model, env, memory=memory, n_episodes=1, random=False, noise=None, render=False)
        global_data.append(params)
        init_data.append(params)
        fitness_start_point.append(fit[0])

    #loading CEM
    print("Loading CEM results...")
    for i in range(NB_RUNS):
        for j in range(CEM_STEP,CEM_END+CEM_STEP,CEM_STEP):
            print("cem : "+str(i)+"_"+str(j))
            model = Actor(state_dim, action_dim, max_action, args)
            model.load_model("cem_explo_2","actor_cem_"+str(i)+"_"+str(j))
            params = model.get_params()
            fit = evaluate(model, env, memory=memory, n_episodes=1, random=False, noise=None, render=False)
            global_data.append(params)
            cem_data.append(params)
            fitness_cem.append(fit[0])
    #loading TD3
    print("Loading TD3 results...")
    for i in range(NB_RUNS):
        for j in range(TD3_STEP,TD3_END+TD3_STEP,TD3_STEP):
            print("td2 : "+str(i)+"_"+str(j))
            model = Actor(state_dim, action_dim, max_action, args)
            model.load_model("td3_explo_2","actor_td3_"+str(i)+"_"+str(j))
            params = model.get_params()
            fit = evaluate(model, env, memory=memory, n_episodes=1, random=False, noise=None, render=False)
            global_data.append(params)
            td3_data.append(params)
            fitness_td3.append(fit[0])

    print("Results loaded")
    print("number of points : "+str(np.array(global_data).shape))

    # Using T-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(global_data)#using global data (all) here

    # Separating the results of each algorithm
    init_tsne_x = tsne_results[0:len(init_data),0]
    init_tsne_y = tsne_results[0:len(init_data),1]
    cem_tsne_x = tsne_results[len(init_data):len(init_data)+len(cem_data),0]
    cem_tsne_y = tsne_results[len(init_data):len(init_data)+len(cem_data),1]
    td3_tsne_x = tsne_results[len(init_data)+len(cem_data):len(init_data)+len(cem_data)+len(td3_data),0]
    td3_tsne_y = tsne_results[len(init_data)+len(cem_data):len(init_data)+len(cem_data)+len(td3_data),1]

    # Computing min and max fitness for colorbar calibration
    min_fitness = min(np.min(fitness_td3), np.min(fitness_cem), np.min(fitness_start_point))
    max_fitness = max(np.max(fitness_td3), np.max(fitness_cem), np.max(fitness_start_point))

    # Plot result  
    plt.scatter(cem_tsne_x , cem_tsne_y , c = fitness_cem, cmap=cm.Blues, vmin=min_fitness, vmax= max_fitness)#CEM
    cb_cem = plt.colorbar()
    cb_cem.set_label("CEM", rotation = 0, fontsize = 13)
    plt.scatter(td3_tsne_x , td3_tsne_y , c = fitness_td3, cmap=cm.Reds, vmin=min_fitness, vmax= max_fitness)#TD3
    cb_td3 = plt.colorbar() 
    cb_td3.set_label("TD3", rotation = 0, fontsize = 13)
    plt.scatter(init_tsne_x , init_tsne_y , c = 'g') #init points
    plt.savefig("resGen_new.png")
    plt.show()

    env.close()
