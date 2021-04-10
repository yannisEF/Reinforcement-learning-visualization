# coding: utf-8

import pickle
import lzma

import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.widgets import Slider
from PIL import Image, ImageDraw

import colorTest
from vector_util import valueToRGB, invertColor, checkFormat


saveFormat = '.xz'
@checkFormat(saveFormat)
def loadFromFile(filename, folder="SavedVignette"):
	"""
	Returns a saved plot
	"""
	with lzma.open(folder+"/"+filename, 'rb') as handle:
		content = pickle.load(handle)
	return content		
		
class SavedVignette:
	"""
	Class storing a Vignette, able to draw it in 3D and 2D
	Useful to serialize in order to be able to change drawing parameters
	"""
	def __init__(self, D, indicesPolicies=None, policyDistance=None,
				 stepalpha=.25, color1=colorTest.color1, color2=colorTest.color2,
				 pixelWidth=10, pixelHeight=10,
				 x_diff=2., y_diff=2.):

		# Content of the Vignette
		self.baseLines = []	# Bottom lines
		self.baseLinesLogProb = [] # log(P(A|S)) for bottom lines
		self.lines = []	# Upper lines
		self.linesLogProb = [] # log(P(A\S)) for upper lines
		self.directions = D	# All sampled directions
		self.indicesPolicies = indicesPolicies # Index of directions that go through a policy
		self.policyDistance = policyDistance # Distance of each policy along its direction
		
		# 2D plot
		self.stepalpha = stepalpha # Distance between each model along a direction
		self.color1, self.color2 = color1, color2 # Min color and max color
		self.pixelWidth, self.pixelHeight = pixelWidth, pixelHeight # Pixels' dimensions

		# 3D plot
		self.fig, self.ax = None, None
		self.x_diff = x_diff #	Distance between each model along a direction
		self.y_diff = y_diff #  Distance between each direction
		
	@checkFormat(saveFormat)
	def saveInFile(self, filename):
		"""
		Save the Vignette in a file
		"""
		with lzma.open(filename, 'wb') as handle:
			pickle.dump(self, handle)
	
	@checkFormat('.png')
	def save2D(self, filename, img=None):
		"""
		Save the Vignette as 2D image
		"""
		img = self.plot2D() if img is None else img
		img.save(filename, format='png')
	
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
								computedImg=None, angles3D=[0], elevs=[0]):
		"""
		Centralises the saving process
		"""
		if saveInFile is True: self.saveInFile(directoryFile+'/'+filename)
		if save2D is True: self.save2D(directory2D+'/'+filename+'_2D', img=computedImg)
		if save3D is True: self.save3D(directory3D+'/'+filename+'_3D', elevs=elevs, angles=angles3D)

	def plot2D(self, color1=None, color2=None, alpha=0):
		"""
		Compute the 2D image of the Vignette

		Cannot store it as PIL images are non serializable
		"""
		color1, color2 = self.color1 if color1 is None else color1, self.color2 if color2 is None else color2
		
		width, height = self.pixelWidth * len(self.lines[-1]), self.pixelHeight * (len(self.lines) + len(self.policyDistance) + len(self.baseLines) + 1)
		newIm = Image.new("RGB",(width, height))
		newDraw = ImageDraw.Draw(newIm)
		meanValue, stdValue = np.mean(self.lines+self.baseLines), np.std(self.lines+self.baseLines)
		minColor, maxColor = meanValue - stdValue, np.max(self.lines+self.baseLines)
		#	Adding the results
		y0 = 0
		for l in range(len(self.lines)):
			# 	Drawing the results
			y1 = y0 + self.pixelHeight
			for c in range(len(self.lines[l])):
				x0 = c * self.pixelWidth
				x1 = x0 + self.pixelWidth

				value = self.lines[l][c] - alpha * self.linesLogProb[l][c]
				color = valueToRGB(value, color1, color2, minNorm=minColor, maxNorm=maxColor)
				newDraw.rectangle([x0, y0, x1, y1], fill=color)
			y0 += self.pixelHeight
			
		# 	Adding the separating line
		y0 += self.pixelHeight
		y1 = y0 + self.pixelHeight
		color = valueToRGB(0, color1, color2, minNorm=minColor, maxNorm=maxColor)
		newDraw.rectangle([0, y0, width, y1], fill=color)

		#	Adding the baseLines (bottom lines)
		for l in range(len(self.baseLines)):
			y0 += self.pixelHeight
			y1 = y0 + self.pixelHeight
			for c in range(len(self.lines[l])):
				x0 = c * self.pixelWidth
				x1 = x0 + self.pixelWidth
				value = self.baseLines[l][c] - alpha * self.baseLinesLogProb[l][c]
				color = valueToRGB(value, color1, color2, minNorm=minColor, maxNorm=maxColor)
				newDraw.rectangle([x0, y0, x1, y1], fill=color)
		
		# 	Adding the policies
		if self.indicesPolicies is not None:
			marginX, marginY = int(self.pixelWidth/4), int(self.pixelHeight/4)
			for k in range(len(self.indicesPolicies)):
				index, distance = self.indicesPolicies[k], round(self.policyDistance[k]/self.stepalpha)
				x0, y0 = (distance + len(self.lines[0])//2) * self.pixelWidth, index * self.pixelHeight
				x1, y1 = x0 + self.pixelWidth, y0 + self.pixelHeight
				color = invertColor(newIm.getpixel((x0,y0)))
				newDraw.ellipse([x0+marginX, y0+marginY, x1-marginX, y1-marginY], fill=color)
				newDraw.text((x0+ int(1.5 * marginX), y0), str(k), fill=invertColor(color))
		
		return newIm

	def plot3D(self, function=lambda x:x,
			   figsize=(12,8), title="Vignette 3D", surfaces=True,
			   alpha=0, minAlpha=.0, maxAlpha=5, transparency=1,
			   **kwargs):
		"""
		Compute the 3D image of the Vignette with surfaces or not, can be shaped by an input function
		"""
		self.fig, self.ax = plt.figure(title,figsize=figsize), plt.axes(projection='3d')

		# Computing the intial 3D Vignette
		if surfaces is True:
			args = [function]
			
			# Default key word arguments
			if "width" not in kwargs.keys():	kwargs["width"] = 5
			if "linewidth" not in kwargs.keys():	kwargs["linewidth"] = .01
			if "cmap" not in kwargs.keys():	kwargs["cmap"] = "coolwarm"
		else:
			args = [function]
				
		self.computeFunction(*args, alpha=alpha, transparency=transparency, surfaces=surfaces, **kwargs)
			
		# Making a slider to allow to change alpha
		axEntropy = plt.axes([0.2, 0.1, 0.65, 0.03])
		self.entropySlider = Slider(ax=axEntropy, label="Alpha", valmin=minAlpha, valmax=maxAlpha, valinit=alpha)
		
		# Making a slider to change the transparency of the surfaces
		axTrans = plt.axes([0.2, 0.06, 0.65, 0.03])
		self.transSlider = Slider(ax=axTrans, label="Transparency", valmin=0, valmax=1, valinit=transparency)
		
		# Update functions
		#	Entropy
		def updateEntropy(val):
			self.ax.clear()
			self.computeFunction(*args, self.entropySlider.val, transparency=self.transSlider.val, surfaces=surfaces, **kwargs)
			self.fig.canvas.draw_idle()
		self.entropySlider.on_changed(updateEntropy)
		
		#	Transparency
		if surfaces is True:
			def updateTrans(val):
				self.ax.clear()
				self.computeFunction(*args, self.entropySlider.val, transparency=self.transSlider.val, surfaces=surfaces, **kwargs)
				self.fig.canvas.draw_idle()
			self.transSlider.on_changed(updateTrans)
				
	def computeFunction(self, function, alpha=1, transparency=1, surfaces=True,
					    width=0, linewidth=0, cmap="coolwarm"):
		"""
		Function called by the slider
		"""
		# Iterate over all lines
		for step in range(-1, len(self.directions)):
			# Check if current lines is a baseLine
			if step == -1:
				# baseLines are at the bottom of the image
				height = -len(self.directions)-1
				line = [self.baseLines[0][k] - alpha * self.baseLinesLogProb[0][k] for k in range(len(self.baseLines[0]))]
			else:
				# Vignette reads from top to bottom
				height = -step
				line = [self.lines[step][k] - alpha * self.linesLogProb[step][k] for k in range(len(self.lines[step]))]
			
			transformedLine = function(line)
			
			# We have to iterate over all input policies at each step for an easier retrieval of parameters
			if self.indicesPolicies is not None:
				for k in range(len(self.indicesPolicies)):
					if self.indicesPolicies[k] == step:
						self.makePolicy3D(k, height, transformedLine,
										  width=width, linewidth=linewidth)
			
			if surfaces is True:
				x_line = np.linspace(-len(line)/2, len(line)/2, len(line))
				y_line = height * width * np.ones(len(line))

				X = np.array([x_line, x_line])
				Y = np.array([y_line, y_line + width])

				Z = np.array([transformedLine, transformedLine])

				self.ax.plot_surface(self.x_diff * X, self.y_diff * Y, Z, cmap=cmap, linewidth=linewidth, alpha=transparency)
			
			else:
				x_line = np.linspace(-len(line)/2, len(line)/2, len(line))
				y_line = np.ones(len(line))

				self.ax.plot3D(self.x_diff * x_line, self.y_diff * height * y_line, transformedLine)
		
		# Plotting user information
		#	Sampled policies
		self.ax.set_xlabel("Sampled policies")
		posits = [self.x_diff * step for step in range(-len(self.lines[0])//2+1, 0, 5)] \
			   + [self.x_diff * step for step in range(0, len(self.lines[0])//2+1, 5)]
		values = list(range(-len(self.lines[0])//2+1, 0, 5)) \
			   + list(range(0, len(self.lines[0])//2+1, 5))
		self.ax.set_xticks(posits)
		self.ax.set_xticklabels(values)
		
		#	Sampled directions
		self.ax.set_ylabel("Sampled directions")
		if surfaces is True:
			posits = [self.y_diff * (round(width/2) - step * width) for step in range(len(self.directions))] \
				   + [self.y_diff * (round(width/2) - (len(self.directions) + step + 1) * width) for step in range(len(self.baseLines))]
		else:
			posits = [self.y_diff * (-step) for step in range(len(self.directions))] \
				   + [self.y_diff * (-(len(self.directions) + step + 1)) for step in range(len(self.baseLines))]
		values = list(range(1, len(self.lines)+1)) \
			   + list(range(1, len(self.baseLines)+1))
		self.ax.set_yticks(posits)
		self.ax.set_yticklabels(values)
		
		# 	Reward
		self.ax.set_zlabel("Reward")
		
	def makePolicy3D(self, index, height, line,
					 width=0, linewidth=0, textMargin=(.02,.02,.02)):
		"""
		Plot policies input points on the savedVignette's 3D plot
		"""
		distance = round(self.policyDistance[index]/self.stepalpha) # Rounding error ?
		dy = width + round(width/2) if width != 0 else 1
		x, y, z = self.x_diff * distance, self.y_diff * (height * dy), line[round(len(line)//2) + distance]
		self.ax.scatter(x, y, z, marker='x')
		
		mX, mY, mZ = textMargin
		self.ax.text(x + mX, y + mY, z + mZ, s = str(index))
		
	def show2D(self, img=None, color1=None, color2=None):
		color1, color2 = self.color1 if color1 is None else color1, self.color2 if color2 is None else color2
		img = self.plot2D(color1, color2) if img is None else img
		img.show()
	
	def show3D(self):
		plt.show()
	
	def changeColors(self, color1=None, color2=None):
		self.color1 = color1 if color1 is not None else self.color1
		self.color2 = color2 if color2 is not None else self.color2
		
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
	
	# Updating the color palette
	loadedVignette.changeColors(color1=colorTest.color1, color2=colorTest.color2)
	
	# Processing the 2D plot
	print("Processing the 2D plot...")

	for alpha in (0,):
		img = loadedVignette.plot2D(alpha=alpha)
		#loadedVignette.save2D("Vignette_output/Entropy"+args.filename+"_" + str(alpha) + "_2D", img=img)

	#loadedVignette.show2D(img=img)

	
	# Processing the 3D plot
	print("Processing 3D plot...")
	def f(x, ecart=1):
		x = (np.array(x) - np.max(x)) / ecart
		y = np.sinc(x)
		invR = 1 / np.sqrt(x**2 + y**2)
		return invR
	def g(x, ecart=1):
		x = np.array(x) - np.mean(x)
		y1 = np.sinc((x - np.max(x)) / ecart)
		y2 = np.sinc((x - np.min(x)) / ecart)
		invR = np.sign(x) / np.sqrt(x**2 + (y1+y2)**2)
		return invR

	
	#angles, elevs = [45, 80, 85, 90], [0, 30, 89, 90]	
	#loadedVignette.plot3D(title="Surface sans transformation")
	loadedVignette.plot3D(surfaces=False, title="Surface sans transformation")
	#loadedVignette.save3D(filename="Vignette_output/no_tranform", angles=angles, elevs=elevs)
	#loadedVignette.plot3DBand(function=g, width=10, title="Surface isolant les maxs")
	#loadedVignette.save3D(filename="Vignette_output/max_isolated", angles=angles, elevs=elevs)
	# 	Showing the 3D plot
	loadedVignette.show3D()
