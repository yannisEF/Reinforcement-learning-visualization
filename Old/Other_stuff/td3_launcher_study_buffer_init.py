# !/usr/bin/env python3
import argparse
from copy import deepcopy

import gym
import gym.spaces
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpg import DDPG
from td3 import TD3
from models import RLNN
from random_process import *
from util import *
from memory import Memory, SharedMemory

save_average_mu_in_csv = True #enregistrer les différents mu average de l'actor dans un csv pendant l'apprentissage

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs
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

    for ep in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

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


def train(run, n_episodes, output=None, debug=False, render=False):
    """
    Train the whole process
    """

    total_steps = 0
    step_cpt = 0
    n = 0
    df = pd.DataFrame(columns=["total_steps", "average_score", "best_score"] +
                      ["score_{}".format(i) for i in range(args.n_actor)])

    fit_plot = []
    iter_plot = []
    while total_steps < args.max_steps:

        random = total_steps < args.start_steps
        actor_steps = 0

        # training the agent
        f, s = evaluate(agent.actor, env, n_episodes=n_episodes,
                        noise=a_noise, random=random, memory=memory, render=render)
        actor_steps += s
        total_steps += s
        step_cpt += s

        # print score
        prCyan('noisy RL agent fitness:{}'.format(f))

        agent.train(actor_steps)

        # saving models and scores
        if step_cpt >= args.period:
            fit_plot.append(f)
            iter_plot.append(total_steps)
            step_cpt = 0

            f, _ = evaluate(agent.actor, env, n_episodes=args.n_eval)
            prRed('Actor Fitness:{}'.format(f))
            #save fitness :
            if(save_average_mu_in_csv):
                #on sauvegarde au fur-et-à-mesure les Mu moyen obtenus pendant l'apprentissage dans un csv
                save_mu = open("buffer_init_td3_1000.csv", 'a')
                save_mu.write(str(total_steps)+","+str(f)+"\r\n")#+ " : "+str(es.mu))
                save_mu.close

            df.to_pickle(output + "/log.pkl")
            res = {"total_steps": total_steps, "score": f}
                #if args.save_all_models:
            #os.makedirs(output + "/td3_run_{}_{}_steps".format(str(1+run),total_steps),exist_ok=True)
            agent.actor.save_model("actors_buffer", "actor_td3_buffer_"+str(run)+"_"+str(total_steps)+"_gamma="+str(args.discount)+"_100_remplissages")
            step_cpt = 0
            print(res)


        # printing iteration resume
        if debug:
            prPurple('Iteration#{}: Total steps:{} \n'.format(
                n, total_steps))

    return fit_plot, iter_plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='Swimmer-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=1, type=float)#0.9999
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

    # Parameter noise parameters
    parser.add_argument('--param_init_std', default=0.01, type=float)
    parser.add_argument('--param_scale', default=0.2, type=float)
    parser.add_argument('--param_adapt', default=1.01, type=float)

    # Training parameters
    parser.add_argument('--n_actor', default=1, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--n_eval', default=10, type=int)#10
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--max_steps', default=200000, type=int)#200000
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--nbRuns', default=1,type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_false')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)

    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # The environment
    env = gym.make(args.env)
    print("episodes max_steps: "+str(env._max_episode_steps))
    #env._max_episode_steps = 1500
    #print("episodes max_steps: "+str(env._max_episode_steps))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    step_runs_plot = []
    nb_runs = 3
    for run in range(1,1+nb_runs):
        # replay buffer
        memory = Memory(args.mem_size, state_dim, action_dim) #before loop, to have a loaded replay buffer at second step

        # Random seed
        if args.seed > 0:
            np.random.seed(args.seed)
            env.seed(args.seed)
            torch.manual_seed(args.seed)

        # replay buffer
        #memory = Memory(args.mem_size, state_dim, action_dim)

        # action noise
        if args.ou_noise:
            a_noise = OrnsteinUhlenbeckProcess(
                action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)
        else:
            a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

        # agent
        if args.use_td3:
            print("RUNNING : TD3")
            #TD3
            agent = TD3(state_dim, action_dim, max_action, memory, args)
        else:
            print("RUNNING : DDPG")
            #DDPG
            agent = DDPG(state_dim, action_dim, max_action, memory, args)

        if args.mode == 'train':
            print("initializing memory...")
            score = 0
            obs = deepcopy(env.reset())
            done = False
            action = env.action_space.sample()
            n_obs, reward, done, info = env.step(action)
            done_bool =  False
            # adding in memory
            for i in range(1000):
                memory.add((obs, n_obs, action, reward, done_bool))
            env.reset()
            print("memory initialized :",memory.full)
            y,x = train(str(run),n_episodes=args.n_episodes, output=args.output, debug=args.debug, render=False)#modif en brut
            step_runs_plot.append((x,y))
        else:
            raise RuntimeError('undefined mode {}'.format(args.mode))

    env.close()
    plt.figure()
    for run in range(len(step_runs_plot)):
        plt.plot(step_runs_plot[run][0],step1_runs_plot[run][1], color ='b')
    plt.show()
    plt.savefig("resMemory2_gamma="+str(args.discount)+"_1000_remplissages/result.png")
