import numpy as np

class transformFunction:
	"""
	Class containing a function and its parameters.
	Made to transform a 3D Vignette in real time with a slider. 
	
	function must return an array only
	"""
	def __init__(self, function, parameters={"nom":{'value':None, 'minValue':None, 'maxValue':None}}):
		self.function = function
		self.parameters = parameters
		self.initParameters = {k:v for k,v in parameters.items()}
	
	def changeValue(self, name, newValue):
		self.parameters[name]['value'] = newValue
		
	def transform(self, x):
		kwargs = {k:v['value'] for k,v in self.parameters.items()}
		return self.function(x, **kwargs)
	
	def reset(self):
		self.parameters = {k:v for k,v in self.initParameters.items()}

# Identity transform
transformIdentity = transformFunction(lambda x:x, parameters={})

## An example
def isolateExtrema(x, amp=1, spread=1):
	"""
	Function designed to isolate extrema values of an array
	"""
	x = np.array(x) - np.mean(x)
	y1 = np.sinc((x - np.max(x)) / (np.std(x) ** spread))
	y2 = np.sinc((x - np.min(x)) / (np.std(x) ** spread))
	invR = np.sign(x) / np.sqrt(x**2 + (y1+y2)**2)
	return amp * invR

# Enter all modifiable parameters
parameters = {"amp":{"value":1, "minValue":0, "maxValue":5},
			  "spread":{"value":1, "minValue":1, "maxValue":2}}
transformIsolate = transformFunction(isolateExtrema, parameters)
