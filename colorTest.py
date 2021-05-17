# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import argparse

from progress.bar import Bar
from PIL import Image
from PIL import ImageDraw
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from vector_util import *
from slowBar import SlowBar

# Chosen color palette
color1, color2 = (0,0,120), (255,151,0)

@checkFormat('.png')
def createPalette(filename="color_palette",
				  pixelHeight=360, pixelWidth=10, length = 80,
				  color1=(50,0,200), color2=(150,100,0)):
	"""
	Draws the gradient of color between color1 and color2
	"""
	colors = np.linspace(-1,1,length)

	output = Image.new("RGB",(length * pixelWidth, pixelHeight))
	outputDraw = ImageDraw.Draw(output)
	
	for k in range(len(colors)):
		outputDraw.rectangle([k*pixelWidth,0,(k+1)*pixelWidth, pixelHeight], fill=valueToRGB2colors(colors[k], color1, color2, minNorm=-1, maxNorm=1))
		# outputDraw.rectangle([k*pixelWidth,0,(k+1)*pixelWidth, pixelHeight], fill=valueToRGB3colors(colors[k], color1, color3=color2, minNorm=-1, maxNorm=1))
	
	textContent = "Color 1: " + str(color1) + "\nColor 2: " + str(color2)
	outputDraw.text((0,0), textContent, fill=invertColor(color1))

	output.save(filename, format='png')

if __name__ == "__main__":
	createPalette(filename="color_palette", color1=color1, color2=color2)
