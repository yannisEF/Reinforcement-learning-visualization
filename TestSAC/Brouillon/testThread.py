from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from vector_util import getPointsDirection

from slowBar import SlowBar

def multiprocessing_func(modelLoc, env, eval_maxiter, init_score,
             directions, modelDirection, min_step, max_step,
             theta0, num_params, minalpha, maxalpha, stepalpha): 
    
    model = SAC.load("{}/{}".format(*modelLoc))
    d = modelDirection
    
    for step in range(min_step,max_step):
        print("\nDirection ", step, "/", len(directions)-1)
        # Get the next direction
        if step != -1:  d = directions[step]
        # New parameters following the direction
        theta_plus, theta_minus = getPointsDirection(theta0, num_params, minalpha, maxalpha, stepalpha, d)

        with SlowBar('Evaluation along the direction' + str(step), max=len(theta_plus)) as bar:
            # Evaluate using new parameters
            scores_plus, scores_minus = [], []
            for param_i in range(len(theta_plus)):
                print(step)
                # 	Go forward in the direction
                model.policy.load_from_vector(theta_plus[param_i])
                #		Get the new performance
                scores_plus.append(evaluate_policy(model, env, n_eval_episodes=eval_maxiter, warn=False)[0])
                # 	Go backward in the direction
                model.policy.load_from_vector(theta_minus[param_i])
                #		Get the new performance
                scores_minus.append(evaluate_policy(model, env, n_eval_episodes=eval_maxiter, warn=False)[0])
                bar.next() 
        # Inverting scores for a symetrical Vignette (theta_minus going left, theta_plus going right)
        scores_minus = scores_minus[::-1]
        line = scores_minus + [init_score] + scores_plus
        # 	Adding the line to the image
