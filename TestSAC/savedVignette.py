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
				plt.savefig(filename+'_e{}_a{}'.format(elev,angle), format='png')
	
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

	def plot3D(self, figsize=(12,8)):
		"""
		Compute the 3D image of the Vignette
		"""
		self.fig, self.ax = plt.figure(figsize=figsize), plt.axes(projection='3d')
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

			self.ax.plot3D(self.x_diff * x_line, self.y_diff * height * y_line, line)

	def show2D(self):
		self.plot2D()
		plt.imshow(self.final_image, vmin=self.v_min_fit, vmax=self.v_max_fit)		
	def show3D(self):
		self.plot3D()
		plt.show()
		
if __name__ == "__main__":
	print("Parsing arguments")
	parser = argparse.ArgumentParser()

	parser.add_argument('--directory', default="SavedVignette", type=str) # directory containing the savedModel
	parser.add_argument('--filename', default="save1", type=str) # name of the file to load

	args = parser.parse_args()

	# Loading the Vignette
	loadedVignette = loadFromFile(args.filename, folder=args.directory)
	# Closing previously plotted figures
	plt.close()
	# Showing the 2D plot
	loadedVignette.show2D()
	# Processing the 3D plot
	loadedVignette.plot3D()
	# 	Saving the 3D in different angles
	loadedVignette.save3D("Vignette_output/test.png", angles = [45], elevs = [0, 90])
	# 	Showing the 3D plot
	loadedVignette.show3D()
