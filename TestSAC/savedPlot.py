import pickle
import matplotlib.pyplot as plt


def loadFromFile(filename, folder="Saved_plot"):
	"""
	Returns a saved plot
	"""
	with open(folder+"/"+filename, 'rb') as handle:
		content = pickle.load(handle)
	return content

class SavedPlot:
	def __init__(self, axes={}, lines={}, **kwargs):
		# Dic of all the lines and axes of the plot for replotting
		# 	Step of the line:Content of the line
		self.axes = axes
		self.lines = lines
		
		# Dictionnary of plotting parameters
		self.params = kwargs
		
		# Default parameters
		self.default("x_diff", 2)
		self.default("y_diff", 2)
		self.default("line_width", 2)
	
	def default(self, varName, defaultValue):
		"""
		Check if varName is in given parameters, otherwise gives it a default value
		"""
		if varName not in self.params.keys(): self.params[varName] = defaultValue

	def saveInFile(self, filename, folder="Saved_plot"):
		"""
		Saves the axes and their plotting arguments in a file
		"""
		with open("{}/{}.pkl".format(folder, filename), "wb") as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def plot3D(self):
		"""
		Plots the saved graph
		"""
		length = len(self.axs[0])
		x_diff, y_diff = self.params["x_diff"], self.params["y_diff"]
		
		x_line, y_line = np.linspace(-length/2,length/2,length), np.ones(length)
		for step, ax in self.axs.items():
			# Inverting y_line because Vignette reads from top to bottom
			height = -step if step != -1 else -len(self.axs.values())-1
			ax.plot3D(self.params["x_diff"] * x_line, self.params["y_diff"] * height * y_line, line,
					  line_width=self.params["line_width"])
