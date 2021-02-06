# Cose bases
Pytorch implementation of CEM-RL: https://arxiv.org/pdf/1810.01222.pdf

To reproduce the results of the paper:

Without importance mixing:
```console
python es_grad.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

With importance mixing:
```console
python es_grad_im.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

TD3:
```console
python distributed.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

# Update about Swimmer
We observe that Swimmer is the only environment where evolutionnary algorithms perform better than Deep reinforcement learning algorithms.

Maybe it is due to a deceptive gradient.

We will try to resolve these questions.

## First Step: Compare CEM and TD3
We need to compare the behaviour of swimmer after executing TD3 and CEM to check if different strategies are used to move forward.
The files containing our algorithms are ```td3_launcher.py``` and ```cem_launcher.py```. 
We modified a bit the original code to compile, to print data we want and to branch the trained Actor to the environment and run episodes on it, with video output.

Here is our actors Zoo of the different observed comportments : https://72indexdescartes.wixsite.com/swimmer

## Videos of swimmer moving
#### After 100.000 steps of CEM, here is the result (actor fitness 300) :

![CEM result (fitness 300)](demo/swimmer_cem_100000_400px.gif)

#### After 20.000 steps of CEM, here is the result we name "U" (actor fitness 31) : 

![CEM result (fitness 31)](demo/swimmer_cem_10000_400px.gif)

#### Here is the result we name "Rampeur" (actor fitness 42) : 

![CEM result rampeur](demo/Swimmer_Rampeur_1.gif)

#### Here is a good "Rampeur" (actor fitness 75) :

![CEM result rampeur2](demo/Swimmer_Rampeur.gif)

## Major evolution steps
#### Evolution steps observed on multiple 500.000 steps runs of CEM :
![CEM evolution steps](demo/CEM_frise.png)


#### After 5.000 steps of TD3, here is the result (actor fitness 25) : 

![TD3 result (fitness 25)](demo/swimmer_td3_5000_25.gif)

#### After 55.000 steps of TD3, here is the result (actor fitness 29) : 

![TD3 result (fitness 29)](demo/swimmer_td3_55000_29.gif)

#### After 200.000 steps of TD3, here is the result (actor fitness 41) : 

![TD3 result (fitness 41)](demo/swimmer_td3_200000_41.gif)

#### After 250.000 steps of TD3, here is the result (actor fitness 43) : 

![TD3 result (fitness 43)](demo/swimmer_td3_250000_43.gif)

#### After  150.000 steps of TD3, here is the result (actor fitness 55) : 

![TD3 result (fitness 55)](demo/swimmer_td3_150000_55.gif)

## Major evolution steps
#### Evolution steps observed on multiple 300.000 steps runs of TD3 :
![TD3 evolution steps](demo/TD3_frise.png)

##### (Fitness values indicated are only approximations).
