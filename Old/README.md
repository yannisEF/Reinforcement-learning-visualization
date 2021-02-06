# Swimmer environment study

## Tools developed

We developed two tools to help understanding the failure of RL algorithms on Swimmer benchmark : the "Vignette" and the "Gradient study".
The python code can be found in directory "Tools_developed".
Those two scripts are based on CEM-RL code from Aloïs Pourchot.
The "Vignette" tool is using random perturbation method to visualize the geometry of the objective function around an actor.
The "Gradient study" tool displays the geometry of objective function on the direction followed by the algorithm, so the estimated gradient in case of TD3. 
Dot product indication helps to understand the behavior of the algorithm. See the paper (https://github.com/DevMaelFranceschetti/PAnd_Swimmer/blob/master/paper_study_Swimmer.pdf) for more details and explanations. You can see some examples of results obtained with these tools in "Output_examples".
  
Those tools needs Mujoco and Swimmer to be installed, and the code based on CEM-RL to be present in the same directory than the tools.  
  
Another version of the "Vignette" and the "Gradient study" adapted for Atari environments is available, see this repo : https://github.com/DevMaelFranceschetti/Atari

### Vignette parameters

	--env, default='Swimmer-v2', type=str :  benchmark used
	--nb_lines, default=60, type=int : number of directions generated, good values : precise 100, less precise 60 or 50
	--minalpha, default=0.0, type=float : start value for alpha, good value : 0.0
	--maxalpha, default=10, type=float : end value for alpha, good value : large 120 or 100, around actor only 10
	--stepalpha, default=0.25, type=float, step for alpha in the loop, good value : precise 0.25 or 0.5 , less precise 2 or 3
	--eval_maxiter, default=1000, type=float : number of steps for the evaluation. Depends on benchmark used.
	--min_colormap, default=-10, type=int : min score value for colormap used (depend of benchmark used)
	--max_colorma, default=360, type=int : max score value for colormap used (depend of benchmark used)
	--basename, default="actor_td3_2_step_1_", type=str : base (files prefix) name of the actor pkl files to load
	--min_iter, default=1000, type=int : iteration number (file suffix) of the first actor pkl files to load
	--max_iter, default=200000, type=int : iteration number (file suffix) of the last actor pkl files to load
	--step_iter, default=1000, type=int : iteration number between two consecutive actor pkl files to load
	--base_output_filename, default="vignette_output", type=str : name of the output file to create
	--filename, default="TEST_5", type=str : name of the directory containing the actors pkl files to load
  
### Gradient study parameters

	--env, default='Swimmer-v2', type=str)
	--minalpha, default=0.0, type=float : start value for alpha, good value : 0.0
	--maxalpha, default=10, type=float : end value for alpha, good value : 10
	--stepalpha, default=0.5, type=float : step for alpha in the loop, good value : 0.25 ou 0.5
	--eval_maxiter, default=1000, type=float : number of steps for the evaluation. Depend on benchmark used.
	--min_colormap, default=-10, type=int : min score value for colormap used (depend of benchmark used)
	--max_colormap, default=360, type=int : max score value for colormap used (depend of benchmark used)
	--filename, default="TEST_5", type=str : name of the directory containing the actors pkl files to load
	--basename, default="actor_td3_2_step_1_", type=str : base (files prefix) name of the actor pkl files to load
	--min_iter, default=1000, type=int : iteration (file suffix) of the first actor pkl files to load
	--max_iter, default=201000, type=int : iteration (file suffix) of the last actor pkl files to load
	--step_iter, default=1000, type=int : iteration step between two consecutive actor pkl files to load
	--output_filename, default="gradient_output.png", type=str : name of the output file to create

### Use Vignette.py

You can check the comments in the code to understand all the parameters used.
Suppose you have a policy parameters file in a directory "my_parameters" beside the python code, and the policy parameters filename is "params1" (a pkl file), just run :

	python3.6 Vignette.py --env Swimmer-v2 --filename my_parameters --basename params --min_iter 1 --max_iter 1 --step_iter 1 

If you want to run vignette for files "params1" and "params5" for example, you can run :  

	python3.6 Vgnette.py --env Swimmer-v2 --filename my_parameters --basename params --min_iter 1 --max_iter 5 --step_iter 4 
  
If you want to compute more or less directions around the parameters, you can tune the nb_lines parameter. You can also decrease the precision and the number of parameters tested around by increasing the stepalpha parameter. You can also change the maximum distance to the studied parameters by changing the maxalpha value. See an example :  

	python3.6 Vignette.py --env Swimmer-v2 --filename my_parameters --basename params --min_iter 1 --max_iter 1 --step_iter 1 --nb_lines 30 --stepalpha 5 --maxalpha 120  
	
Note that reducing maxalpha, reducing nb_lines, and increasing stepalpha reduces the computation time.

### Use GradientStudy.py

Parameters are quite similar for GradientStudy.py :
Suppose you have 5 policy parameters files in a directory "my_parameters" beside the python code, and each pkl policy parameters filename is "params" followed by the iteration number (ex : params10, params20 ... params50).  
You can run :  

	python3.6 GradientStudy.py --env Swimmer-v2 --filename my_parameters --basename params --min_iter 10 --max_iter 50 --step_iter 10


## TD3 modification

We are proposing a modification of the initial settings of TD3 to achieve good performance on Swimmer: 
use discount = 1 (or very close to 1) and use start_steps = 20000, removing the update from TD3 before start_steps iterations.
See the paper for more details and explanations.
