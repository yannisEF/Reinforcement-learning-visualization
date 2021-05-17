# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from progress.bar import Bar

def checkFormat(fileExt):
	"""
	Checks the format of the input name, modifies it if need be
	"""
	def decorator(function):
		def wrapper(*args, **kwargs):
			# Check if names in args or kwargs
			indexNames, inputNames = None, None
			try:
				inputNames = kwargs['filename']
				indexNames = 'dic'
			except KeyError:
				for i in range(len(args)):
					if type(args[i]) in [list, str]: indexNames = i;	inputNames = args[i];	break
			if indexNames is None:	raise NameError("No file as parameters")
			
			# Add the desired file extension to the filenames
			if type(inputNames) is not list:	changedInput = [inputNames]
			else:	changedInput = inputNames[:]
			changedInput = [f+fileExt if f[-len(fileExt):] != fileExt else f for f in changedInput]
			if type(inputNames) is not list:	changedInput = changedInput[0]
			
			# Change the function's parameters
			if indexNames == 'dic':	kwargs['filename'] = changedInput
			else:	args = list(args[:indexNames]) + [changedInput] + list(args[indexNames+1:])
			return function(*args, **kwargs)
		return wrapper
	return decorator	

def valueToRGB(*args, **kwargs):
	return valueToRGB3colors(*args,**kwargs)
	
def valueToRGB2colors(value, color1=(255,0,0), color2=(0,255,0), pureNorm=None, minNorm=-1, maxNorm=1):
	"""
	Converts a value to an RGB color, between color1 and color2
	Pure colors for values of norm >= pureNorm
	"""
	if pureNorm is not None:
		if value**2 > pureNorm**2:
			value = pureNorm if value > 0 else -pureNorm
		weight = value/pureNorm
		return tuple(int(color1[k] * (1-weight)/2) + int(color2[k] * (1+weight)/2) for k in range(len(color1)))
	
	value = minNorm if value <= minNorm else value
	value = maxNorm if value >= maxNorm else value
	
	weight1, weight2 = abs((maxNorm - value)/(maxNorm - minNorm)), abs((minNorm - value)/(maxNorm - minNorm))
	return tuple(int(color1[k] * weight1) + int(color2[k] * weight2) for k in range(len(color1)))

def valueToRGB3colors(value, color1=(255,0,0), color2=None, color3=(0,0,255), pureNorm=None, minNorm=-1, maxNorm=1):
	"""
	Converts a value to an RGB color, between color1, color2 and color3
	Pure colors for values of norm >= pureNorm
	"""
	if color2 is None:
		color2 = tuple(int((color1[k]+color3[k])/2) for k in range(3))
		
	middle = (maxNorm + minNorm) / 2
	if value <= middle:
		return valueToRGB2colors(value, color1, color2, pureNorm=pureNorm, minNorm=minNorm, maxNorm=middle)
	else:
		return valueToRGB2colors(value, color2, color3, pureNorm=pureNorm, minNorm=middle, maxNorm=maxNorm)
	
def invertColor(color):
	"""
	Inverts an RGB color
	"""
	return tuple((255 - p for p in color))

def getPointsChoice(init_params,num_params, minalpha, maxaplha, stepalpha, prob):
	"""
	# Params :

	init_params : model parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	prob : the probability to choose each parameter dimension (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by random choice of proba 'prob' on param dimensions.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives a good but very noisy visualisation and not easy to interpret.
	"""
	#init_params = np.copy(base_params)
	d = np.random.choice([1, 0], size=(num_params,), p=[prob, 1-prob]) #select random dimensions with proba 
	print("proportion: "+str(np.count_nonzero(d==1))+"/"+str(num_params))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniform(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :

	init_params : model parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [0,1).
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives the best visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(0, 1, num_params) #select uniformly dimensions [0,1)
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsDirection(init_params,num_params, minalpha, maxaplha,stepalpha, d):
	"""
	# Params :

	init_params : model parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	d : pre-choosend direction
	
	# Function:

	Returns parameters around base_params on direction given in parameters.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives an output that is comparable with other results if directions are the same.
	"""
	#init_params = np.copy(base_params)
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniformCentered(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :

	init_params : model parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:

	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [-1,1].
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha. 
	This method gives bad visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(-1, 1, num_params) #select uniformly dimensions in [-1,1)
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getDirectionsMuller(nb_directions,num_params):
    """
    # Params :

    nb_directions : number of directions to generate randomly in unit ball
    num_params : dimensions of the vectors to generate (int value, only 1D vectors)
	
    # Function:

    Returns a list of vectors generated in the uni ball of 'num_params' dimensions, using Muller
    """
    D = []
    with Bar('Directions computed', max=nb_directions) as bar:
        for _ in range(nb_directions):
            u = np.random.normal(0,1,num_params)
            norm = np.sum(u**2)**(0.5)
            r = np.random.random()**(1.0/num_params)
            x = r*u/norm

            D.append(x)
		    
            bar.next()
    return D

def euclidienne(x,y):
    """
    # Params :

	
    # Function:

    Returns a simple euclidian distance between x and y.
    """
	
    return np.linalg.norm(np.array(x)-np.array(y))

def order_all_by_proximity(vectors):
    """
    # Params :

    vectors : a list of vectors
	
    # Function:

    Returns the list of vectors ordered by inserting the vectors between their nearest neighbors
    """
    ordered = []
    with Bar('Ordering them between nearest neighbors', max=len(vectors)) as bar:
        for vect in vectors:
            if(len(ordered)==0):
                ordered.append(vect)
            else:
                ind = compute_best_insert_place(vect, ordered)
                ordered.insert(ind,vect)
            bar.next()
    return ordered

def compute_best_insert_place(vect, ordered_vectors):
    """
    # Params :

    ordered_vectors : a list of vectors ordered by inserting the vectors between their nearest neighbors
    vect : a vector to insert at the best place in the ordered list of vectors
	
    # Function:

    Returns the index where 'vect' should be inserted to be between the two nearest neighbors using euclidien distance
    """
    # Compute the index where the vector will be at the best place :
    value_dist = euclidienne(vect, ordered_vectors[0])
    dist_place = [value_dist]
    for ind in range(len(ordered_vectors)-1):
        value_dist = np.mean([euclidienne(vect, ordered_vectors[ind]),euclidienne(vect, ordered_vectors[ind+1])])
        dist_place.append(value_dist)
    value_dist = euclidienne(vect, ordered_vectors[len(ordered_vectors)-1])
    dist_place.append(value_dist)
    ind = np.argmin(dist_place)
    return ind
