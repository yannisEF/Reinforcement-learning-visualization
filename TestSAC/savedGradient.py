import colorTest

import pickle
import lzma
import argparse
import numpy as np

from PIL import Image
from PIL import ImageDraw

from vector_util import valueToRGB, invertColor, checkFormat

saveFormat = '.xz'
@checkFormat(saveFormat)
def loadFromFile(filename, folder="SavedGradient"):
	"""
	Returns a saved plot
	"""
	with lzma.open(folder+"/"+filename, 'rb') as handle:
		content = pickle.load(handle)
	return content

class SavedGradient:
    """
    Class storing a GradientStudy, able to compute and draw its image
    Useful to serialize in order to be able to change drawing parameters
    """
    def __init__(self, directions=[], results=[], red_markers=[], green_markers=[],
                nbLines=3, color1=colorTest.color1, color2=colorTest.color2, pixelWidth=20, pixelHeight=10, maxValue=360,
                dotText=True, dotWidth=150, xMargin=10, yMargin=10):
        # Content of the GradientStudy
        self.directions = directions # directions taken by the model at each file
        self.results = results # Contains the results along the directions taken
        self.red_markers, self.green_markers = red_markers, green_markers # Model's positions

        # Image parameters
        self.nbLines = nbLines # The height in number of pixel for each result
        self.color1, self.color2 = color1, color2 # Color palette used by the gradient
        self.pixelWidth, self.pixelHeight = pixelWidth, pixelHeight
        self.maxValue = maxValue
        
        # Dot product parameters
        self.dotText = dotText # True if we want to show the value of the dot product
        self.dotWidth = dotWidth # Width of the side panel containing the dot product
        self.xMargin, self.yMargin = xMargin, yMargin # Margin for the dot product's bar
	
    @checkFormat(saveFormat)
    def saveGradient(self, filename, directory="SavedGradient"):
        """
        Serializes the gradient into a .xz file
        """
        with lzma.open(directory+'/'+filename, 'wb') as handle:
            pickle.dump(self, handle)

    def computeImage(self, color1=None, color2=None,
                     saveImage=True, filename='output', directory="Gradient_output"):
        """
        Computes the image of the gradient and saves it if asked
        """
        if color1 == None or color2 == None: color1, color2 = self.color1, self.color2

        width, height = self.pixelWidth * len(self.results[-1]), self.pixelHeight * len(self.results) * (self.nbLines+1)
        newIm = Image.new("RGB",(width+self.dotWidth, height))
        newDraw = ImageDraw.Draw(newIm)

        meanValue, stdValue = np.mean(self.results), np.std(self.results)
        minColor, maxColor = meanValue - stdValue, np.max(self.results)
        #	Putting the results and markers
        for l in range(len(self.results)):
            #	Separating lines containing the model's markers
            x0, y0 = self.red_markers[l] * self.pixelWidth, l * (self.nbLines+1) * self.pixelHeight
            x1, y1 = x0 + self.pixelWidth, y0 + self.pixelHeight
            newDraw.rectangle([x0, y0, x1, y1], fill=(255,0,0))

            x0 = self.green_markers[l] * self.pixelWidth
            x1 = x0 + self.pixelWidth
            newDraw.rectangle([x0, y0, x1, y1], fill=(0,255,0))

            # 	Drawing the results
            y0 += self.pixelHeight
            y1 = y0 + self.nbLines * self.pixelHeight
            for c in range(len(self.results[l])):
                x0 = c * self.pixelWidth
                x1 = x0 + self.pixelWidth
                color = valueToRGB(self.results[l][c], color1, color2, minNorm=minColor, maxNorm=maxColor)
                newDraw.rectangle([x0, y0, x1, y1], fill=color)

            #	Processing the dot product,
            if l < len(self.results)-1:
                dot_product = np.dot(self.directions[l], self.directions[l+1])
                color = valueToRGB(dot_product, (255,0,0), (0,255,0), pureNorm=1)

                # Putting in on the side with a small margin
                x0, y0Dot = self.xMargin + width, y1 - self.yMargin
                x1, y1Dot = x0 + min(self.dotWidth * abs(dot_product), self.dotWidth-self.xMargin), y1 + self.pixelHeight + self.yMargin
                newDraw.rectangle([x0, y0Dot, x1, y1Dot], fill=color)

                # Showing the value of the dot product if asked
                if self.dotText is True:    newDraw.text((x0 + self.xMargin,y1), "{:.2f}".format(dot_product), fill=invertColor(color))

        # Saving image if asked
        if saveImage is True:   newIm.save(directory+'/'+filename+'.png', format='png')

    def changeColor(self, color1, color2):
        """
        Changes the color palette used in the gradient's image
        """
        self.color1, self.color2 = color1, color2
    
if __name__ == "__main__":
    print("Parsing arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', default="SavedGradient", type=str) # directory containing the savedModel
    parser.add_argument('--filename', default="rl_model_", type=str) # name of the file to load
    
    parser.add_argument('--outputDir', default="Gradient_output", type=str) # output directory
    parser.add_argument('--outputName', default="test_palette", type=str) # output name

    args = parser.parse_args()

    # Loading the gradient
    loadedGradient = loadFromFile(args.filename, folder=args.directory)
    # Changing the color palette
    #color1, color2 = (40,0,200), (120,120,0)
    #loadedGradient.changeColor(color1=color1, color2=color2)
    # Computing the new image and saving the results
    loadedGradient.computeImage(filename=args.outputName, directory=args.outputDir)
