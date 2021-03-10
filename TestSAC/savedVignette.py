# coding: utf-8

import pickle
import lzma

import matplotlib.pyplot as plt
import numpy as np
import argparse

def loadFromFile(filename, folder="SavedVignette"):
	"""
	Returns a saved plot
	"""
	if filename[-3:] != ".xz": filename = filename+'.xz'
	with lzma.open(folder+"/"+filename, 'rb') as handle:
		content = pickle.load(handle)
	return content

class SavedVignette:
	"""
	Class storing a Vignette, able to draw it in 3D and 2D
	Useful to serialize in order to be able to change drawing parameters
	"""
	def __init__(self, d, D, length_dist,
				 v_min_fit=-10, v_max_fit=360, stepalpha=.25, resolution=10,
				 x_diff=2., y_diff=2., line_width=1.):

		# Content of the Vignette
		self.baseLines = []	# Bottom lines
		self.lines = []	# Upper lines
		self.model_direction = d
		self.directions = D	# All sampled directions

		# Image
		self.final_image = None
		#	Drawing parameters
		self.v_max_fit = v_max_fit
		self.v_min_fit = v_min_fit
		self.stepalpha = stepalpha
		self.length_dist = length_dist
		#		Size of each pixel
		self.resolution = resolution

		# 3D plot
		self.fig, self.ax = None, None
		# Drawing parameters
		self.x_diff = x_diff
		self.y_diff = y_diff
		self.line_width = line_width

	def saveInFile(self, filename):
		"""
		Save the Vignette in a file
		"""
		with lzma.open(filename, 'wb') as handle:
			pickle.dump(self, handle)
	
	def save2D(self, filename):
		"""
		Save the Vignette as 2D image
		"""
		plt.imsave(filename, self.final_image,
					vmin=self.v_min_fit, vmax=self.v_max_fit,
					format='png')

	def save3D(self, filename, elevs=[30], angles=[0]):
		"""
		Save the Vignette as a 3D image
		"""
		for elev in elevs:
			for angle in angles:
				self.ax.view_init(elev, angle)
				plt.draw()
				plt.savefig(filename+'_e{}_a{}.png'.format(elev,angle), format='png')
	
	def saveAll(self, filename, saveInFile=False, save2D=False, save3D=False,
								directoryFile="SavedVignette", directory2D="Vignette_output", directory3D="Vignette_output",
								angles3D=[0], elevs=[0]):
		"""
		Centralises the saving process
		"""
		if saveInFile is True: self.saveInFile(directoryFile+'/'+filename+'.xz')
		if save2D is True: self.save2D(directory2D+'/'+filename+'_2D'+'.png')
		if save3D is True: self.save3D(directory3D+'/'+filename+'_3D'+'.png', elevs=elevs, angles=angles3D)

	def plot2D(self):
		"""
		Compute the 2D image of the Vignette
		"""
		width = len(self.baseLines[0])
		separating_line = np.zeros(width)

		self.final_image = np.concatenate((self.lines, [separating_line], self.baseLines), axis=0)
		self.final_image = np.repeat(self.final_image, self.resolution, axis=0)
		self.final_image = np.repeat(self.final_image, self.resolution, axis=1)

	def plot3D(self, function=lambda x:x, figsize=(12,8), title="Vignette ligne"):
		"""
		Compute the 3D image of the Vignette
		"""
		self.fig, self.ax = plt.figure(title,figsize=figsize), plt.axes(projection='3d')
		# Iterate over all lines
		for step in range(-1, len(self.directions)):
			# Check if current lines is a baseLine
			if step == -1:
				# baseLines are at the bottom of the image
				height = -len(self.directions)-1
				line = self.baseLines[0]
			else:
				# Vignette reads from top to bottom
				height = -step
				line = self.lines[step]

			x_line = np.linspace(-len(line)/2, len(line)/2, len(line))
			y_line = np.ones(len(line))

			self.ax.plot3D(self.x_diff * x_line, self.y_diff * height * y_line, function(line))

	def plot3DBand(self, function=lambda x:x,
				   figsize=(12,8), title="Vignette surface", width=5, linewidth=.01, cmap='coolwarm'):
		"""
		Compute the 3D image of the Vignette
		"""
		self.fig, self.ax = plt.figure(title,figsize=figsize), plt.axes(projection='3d')
		# Iterate over all lines
		for step in range(-1, len(self.directions)):
			# Check if current lines is a baseLine
			if step == -1:
				# baseLines are at the bottom of the image
				height = -len(self.directions)-1
				line = self.baseLines[0]
			else:
				# Vignette reads from top to bottom
				height = -step
				line = self.lines[step]
			
			x_line = np.linspace(-len(line)/2, len(line)/2, len(line))
			y_line = height * width * np.ones(len(line))

			X = np.array([x_line, x_line])
			Y = np.array([y_line, y_line + width])

			newLine = function(line)
			Z = np.array([newLine, newLine])

			self.ax.plot_surface(self.x_diff * X, self.y_diff * Y, Z, cmap=cmap, linewidth=linewidth)
			

	def show2D(self, cmap='binary'):
		self.plot2D()
		plt.imshow(self.final_image, vmin=self.v_min_fit, vmax=self.v_max_fit, cmap=cmap)		
	def show3D(self):
		plt.show()
		
if __name__ == "__main__":
	print("Parsing arguments...")
	parser = argparse.ArgumentParser()

	parser.add_argument('--directory', default="SavedVignette", type=str) # directory containing the savedModel
	parser.add_argument('--filename', default="save1", type=str) # name of the file to load

	args = parser.parse_args()

	# Loading the Vignette
	print("Loading the Vignette...")
	loadedVignette = loadFromFile(args.filename, folder=args.directory)
	# Closing previously plotted figures
	plt.close()

	# Showing the 2D plot
	print("Processing 2D plot...")
	loadedVignette.v_min_fit, loadedVignette.v_max_fit = -2000, 0
	loadedVignette.show2D()
	
	# Processing the 3D plot
	print("Processing 3D plot...")
	def f(x, ecart=1):
		x = (np.array(x) - np.max(x)) / ecart
		y = np.sinc(x)
		invR = 1 / np.sqrt(x**2 + y**2)
		return invR
	
	angles, elevs = [45, 80, 85, 90], [0, 30, 89, 90]	
	loadedVignette.plot3DBand(width=10, title="Surface sans transformation")
	#loadedVignette.save3D(filename="Vignette_output/no_tranform", angles=angles, elevs=elevs)
	loadedVignette.plot3DBand(function=f, width=10, title="Surface isolant les maxs")
	#loadedVignette.save3D(filename="Vignette_output/max_isolated", angles=angles, elevs=elevs)
	# 	Showing the 3D plot
	loadedVignette.show3D()
