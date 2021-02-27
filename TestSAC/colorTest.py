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

# Draws the gradient of color between color1 and color2
image_filename = "test_color"
pixelHeight, pixelWidth = 360, 10
length = 80

color1, color2 = (50,0,200), (150,100,0)
if __name__ == "__main__":
	colors = np.linspace(-1,1,length)

	output = Image.new("RGB",(length * pixelWidth, pixelHeight))
	outputDraw = ImageDraw.Draw(output)
	
	for k in range(len(colors)):
		outputDraw.rectangle([k*pixelWidth,0,(k+1)*pixelWidth, pixelHeight], fill=valueToRGB(colors[k], color1, color2))
		
	output.save(image_filename+'2.png', format='png')

	
