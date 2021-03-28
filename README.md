# P_androide

 ## Visualising the value landscape to better understand reinforcement learning

This project consists of developing tools for visualising the value landscape and gradient descent trajectories to better understand the dynamics of reinforcement learning algorithms.
Our tools were applied to the classic **Swimmer** control environment and **OpenAI Gym Pendulum-v0** using Soft Actor Critic (SAC) algorithm,with diffrent alternate slightly changes the policy update rule.

-We use the implementations of reinforcement learning algorithms in PyTorch of : https://github.com/DLR-RM/stable-baselines3


-We provide a visualization tools which can be used straightforwardly and helps understanding the gradient landscape in the parameter space and the trajectory of a deep RL algorithm in that space.


**To train a model** 

"/TestSAC/trainModel.py"


Output of the tool : you will find in "/TestSAC/Models" a ".zip"  file of a trained model

```
python3 trainModel.py --env Pendulum-v0 --tau 0.005 --learning_rate 0.0003 --save_path Models --save_freq 1000

```

**Vignette**   

"/TestSAC/Vignette.py"  calculates the Vignettes of a set of files


Input of the tool :  a ".zip"  file of a trained model


Output of the tool : you will find in "/TestSAC/Vignette_output" images of 2D and 3D Vignettes



You can run tests from the command-line : (~8 minutes computing time) (savedVignette can be very heavy)

```
python3 Vignette.py --env Pendulum-v0 --inputDir Models/Pendulum --min_iter 8000 --max_iter 8000 --step_iter 500 --eval_maxiter 5 --nb_lines 10

```

**GradientStudy**

Input of the tool :  a ".zip"  file of a trained model



Output of the tool :an image showing a gradient study  (look at the example in "/TestSAC/Gradient_output/" )



You can run tests from the command-line :(~5 minutes computing time)

```
python3 GradientStudy.py --env Pendulum-v0 --directory Models/Pendulum --min_iter 500 --max_iter 10000 --step_iter 500 --eval_maxiter 5
```
**To plot 2D & 3D savedVignette results** 

"/TestSAC/savedVignette.py"


Output of the tool : you will find in "/TestSAC/SavedVignette" a ".xz"  file of a savedVignette

```
python3 savedVignette.py --directory cheminVersSavedVignette --filename rl_model_8000_steps

```

## Installation of dependencies

```
 pip install -r requirements.txt

```
