import argparse
import importlib.util
import os

from secml_malware.models import CClassifierEnd2EndMalware, MalConv

from constants import *
from prompts import error_prompt, success_prompt, crash_prompt
from state import global_state


def get_target_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('target',
						help='target for the attacks. Examples already implemented: malconv, gbdt_ember, sorel_dnn')
	parser.add_argument('--model_path',
						help='path to the weights of the model. Leave empty for MalConv to load default weights embedded in library.')
	return parser


def target(args):
	if args.target is None:
		error_prompt('You have to set a target.')
		error_prompt(f'Chose one from this list: {ALL_MODELS}')
		return

	if args.target == MALCONV:
		clf = CClassifierEnd2EndMalware(model=MalConv())
		if args.model_path is None:
			clf.load_pretrained_model()
		else:
			clf.load_pretrained_model(args.model_path)
		_set_target(clf)
		return

	if not os.path.isdir(global_state.plugin_path):
		crash_prompt(f"The plugin path {global_state.plugin_path} does not exists!")
		return

	plugins = os.listdir(global_state.plugin_path)
	if args.target not in plugins:
		error_prompt(f"The target {args.target} does not exists among the plugins.")
		return

	try:
		module = importlib.import_module(f'plugins.{args.target}.model')
		clf = module.load_model(args.model_path)

		# # Train the model with the given train data file
        # if args.train_data is not None:
        #     train_data = load_train_data(args.train_data)
        #     # Train the model with the train data
        #     clf.fit(train_data.features, train_data.labels)

		_set_target(clf)
		return
	except Exception as e:
		crash_prompt(f"Can't import plugin {args.target}")
		crash_prompt(f"Error was: {e}")
		return


def _set_target(clf):
	global_state.target = clf
	success_prompt('Target set!')

def load_train_data(file_path):
    # Assuming the train data file is a CSV file where each row represents a sample,
    # and the last column contains the labels
    
    # Load the CSV file
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    
    # Split the features and labels
    features = data[:, :-1]  # All columns except the last one
    labels = data[:, -1]  # Last column
    
    # Return the features and labels as a tuple or any appropriate data structure
    return features, labels