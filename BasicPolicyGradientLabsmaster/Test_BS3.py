import gym
import numpy as np
from wrappers.perf_writer import PerfWriter
from visu.visu_results import plot_data
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd



def moyenne(filename):
    global result
    f=open(filename,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(float(x.split(' ')[1][0:len(x.split(' ')[1])-2]))
    f.close()
    s=0
    for val in result:
        s=s+val
    return s/len(result)


def mean_files(nb_files):
	X=[]
	for i in range(0,nb_files):
	    filename="data/save/reward_Save_{}.txt".format(i)
	    X.append(moyenne(filename))
	return X
	
def plot(nb_files):
	
	f=open("data/save/duration_Save_0.txt","r")
	lines=f.readlines()
	time=[]
	for x in lines:
	    time.append(x.split(' ')[1])
	f.close()
	#print(time)
	for k in range (1,len(time)):

	    time[k-1]=float(time[k-1])
	    time[k]=float(time[k])
	    time[k] = time[k-1]+time[k]
	#print(time)
	#print(X)
	X=mean_files(nb_files)

	plt.plot(time,X)
	plt.ylabel("reward")
	plt.xlabel("duration")
	plt.show()
	
def learnSAC(nb_learn):
        model = SAC('MlpPolicy',"Swimmer-v2", verbose=1, target_update_interval=250)
        eval_env = gym.make("Swimmer-v2")
        perf=PerfWriter(eval_env)

        for k in range(nb_learn):
            model.learn(total_timesteps=int(1000))
            perf.set_file_name("Save_"+str(k))
            mean_reward, std_reward = evaluate_policy(model, perf, n_eval_episodes=nb_learn, deterministic=True)
            print("mean"+str(mean_reward)+"std_reward"+str(std_reward))

	
	
if __name__ == "__main__":

	nb_learn = 100
    #learnSAC(nb_learn)
	plot(nb_learn)