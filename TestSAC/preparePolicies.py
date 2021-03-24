import argparse
import pickle
import lzma

from stable_baselines3 import SAC
from vector_util import checkFormat

# Prepares for comparison a list of policies from the models entered as parameters
# python3 preparePolicies.py --inputNames "rl_model_7000_steps; rl_model_7500_steps; rl_model_8500_steps; rl_model_9000_steps" --outputName "pendulum_around_8000"

saveFormat = '.xz'
def loadFromFile(filename=[''], folder="Models/Pendulum"):
	"""
	Returns a list of policies
	
	Filenames separated by ' '
	"""
	return [SAC.load('{}/{}'.format(folder, f)) for f in filename]

@checkFormat(saveFormat)
def saveInFile(policies, filename='savedPolicies', folder="ComparePolicies"):
	with lzma.open(folder+'/'+filename, 'wb') as handle:
		pickle.dump(policies, handle)

if __name__ == "__main__":
	print("Parsing arguments")
	parser = argparse.ArgumentParser()

	parser.add_argument('--inputFolder', default='Models/Pendulum', type=str) # Folder containing the input
	parser.add_argument('--inputNames', default="", type=str) # Names of every model to load, separated by '; '
	parser.add_argument('--outputFolder', default='ComparePolicies', type=str) # Folder where the list of policies will be saved
	parser.add_argument('--outputName', default="savedPolicies", type=str) # Name of the file containing the list of policies
	args = parser.parse_args()
	
	print("Retrieving the models..")
	models = loadFromFile(filename=args.inputNames.split('; '), folder=args.inputFolder)
	print("Processing the models' policies..")
	policies = [model.policy.parameters_to_vector() for model in models]
	print("Saving the list of policies..")
	saveInFile(policies, filename=args.outputName, folder=args.outputFolder)
	
	
