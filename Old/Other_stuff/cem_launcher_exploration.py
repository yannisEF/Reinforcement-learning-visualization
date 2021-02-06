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

save_average_mu_in_csv = False #True #enregistrer les différents mu average de l'actor dans un csv pendant l'apprentissage
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor #?


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    #modified version to separate the reward and the cost. For initial evaluate function, refer to evaluate2.
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
    costs = []#to remove for normal run
    steps = 0

    for _ in range(n_episodes):
        cost = 0
        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, info = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            cost += info['reward_ctrl']*1000 #to remove for normal run
            score += info['reward_fwd'] - info['reward_ctrl']*1000 #to remove for normal run  #reward
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
        costs.append(cost)#to remove for normal run

    return np.mean(scores), steps, np.mean(costs)

def evaluate2(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False): #init evaluation
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
    parser.add_argument('--max_steps', default=200000, type=int)
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
    """args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))
    """
    nb_runs_CEM = 100
    #number of runs to do. Warning : the name of the output directory for actor and video depends of this value.
    # Think about changing the start value if you already have output results.
    os.makedirs("cem_explo",exist_ok=True)
    for run in range(55,55+nb_runs_CEM):
        
        print("creating environment")
        # environment
        env = gym.make(args.env)
      
        print("starting...")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = int(env.action_space.high[0])
        step_cpt = 0
        # memory
        memory = Memory(args.mem_size, state_dim, action_dim)
    
        # actor
        actor = Actor(state_dim, action_dim, max_action, args)
        theta0 = actor.get_params()
        num_params = len(theta0)
        random_params = np.random.uniform(0, 1, num_params)
        actor.set_params(random_params)
        actor.save_model("cem_explo" , "actor_cem_"+str(run)+"_"+str(step_cpt))
        actor_t = Actor(state_dim, action_dim, max_action, args)
        actor_t.load_state_dict(actor.state_dict())

        # action noise
        if not args.ou_noise:
            a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
        else:
            a_noise = OrnsteinUhlenbeckProcess(
                action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)

        if USE_CUDA:
            actor.cuda()
            actor_t.cuda()

        # CEM
        es = sepCEM(actor.get_size(), mu_init=actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                    pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
        # es = Control(actor.get_size(), pop_size=args.pop_size, mu_init=actor.get_params())

        # training
        
        total_steps = 0
        actor_steps = 0
        df = pd.DataFrame(columns=["total_steps", "average_score",
                                 "average_score_rl", "average_score_ea", "best_score"])
        while total_steps < args.max_steps:

            fitness = []
            #costs = []
            fitness_ = []
            es_params = es.ask(args.pop_size)
            actor_steps = 0
        

            # evaluate noisy actor(s)
            for i in range(args.n_noisy):
                actor.set_params(es_params[i])
                f, steps = evaluate2(actor, env, memory=memory, n_episodes=args.n_episodes,
                                    render=args.render, noise=a_noise)
                actor_steps += steps
                prCyan('Noisy actor {} fitness:{}'.format(i, f))

            # evaluate all actors
            for params in es_params:

                actor.set_params(params)
                f, steps= evaluate2(actor, env, memory=memory, n_episodes=args.n_episodes, render=args.render)
                actor_steps += steps
                fitness.append(f)
                #costs.append(cost)

                # print scores
                prLightPurple('Actor fitness:{}'.format(f))
                #prLightPurple('Actor fitness:{} cost:{}'.format(f,cost))
        
        
            # update es
            es.tell(es_params, fitness)
            print(es.elite)

            # update step counts
            total_steps += actor_steps
            step_cpt += actor_steps

            # save stuff
            if step_cpt >= args.period:
                actor.set_params(es.mu)#mu
                actor.save_model("cem_explo" , "actor_cem_"+str(run)+"_"+str(total_steps))
                """else:
                    actor.set_params(es.mu)
                    actor.save_model(args.output, "actor")
                """
                step_cpt = 0

            print("Total steps", total_steps)
        env.close()
    """
    #########################################
    ###### TEST DE L'ACTOR sur Swimmer: #####
    #########################################
    #
    # Note : on cherche ici à faire fonctionner le Swimmer en le branchant à l'actor qui a été entrainé.
    # On commence par réinitialiser un environnement Swimmer totalement neutre, puis on appelle
    # l'acteur pour choisir l'action à faire, et on visualise le résultat au fur-et-à-mesure avec render().
    
    #création d'un environnement Swimmer
    env2 = gym.make('Swimmer-v2')
    #ajout d'un moniteur pour l'enregistrement vidéo sur l'environnement :
    env2 = gym.wrappers.Monitor(env2, './video_cost_bonus',force=True)
    #itialisation de l'environnement :
    env2.reset()
    #chargement d'un état neutre (positions à 0, vélocités à 0) :
    env2.set_state(env2.init_qpos, env2.init_qvel)
    #définition d'une action nulle :
    action = [0. , 0.]
    #récupération de la valeur de l'état grâce à l'action nulle (petite triche) :
    state, reward, done, _ = env2.step(action)
    
    #itérations sur les actions choisies par l'acteur entrainté :
    for _ in range(10000):
        #conversion de l'état dans le bon format (FloatTensor) pour la méthode forward de Actor:
        stateTorch = FloatTensor(np.array(state).reshape(-1))
        #choix de l'action par l'acteur entrainté
        action = actor.forward(stateTorch).cpu().data.numpy().flatten()
        #print(action)
        #on effectue cette action et on récupère le nouvel état
        state, reward, done, _ = env2.step(action)
        if(done):
            state = env2.reset()
        #on affiche le rendu visuel
        #env.render()
    env2.close()
    """
            
