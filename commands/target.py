import argparse
import importlib.util
import os

import numpy as np
from secml_malware.models import CClassifierEnd2EndMalware, MalConv, End2EndModel

from constants import *
from prompts import error_prompt, success_prompt, crash_prompt
from state import global_state

import torch
import torch.nn as nn

from libauc.optimizers import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score


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
        model = MalConv()

        model = model.train(True)
        # print("ok step1")
        model = train_model(model, ['//toucanstrike_webserver/static/Rbot-O.7z'], [1.0])
        # print("ok step2")
        # global_state.model = model
        clf = CClassifierEnd2EndMalware(model=model)
        if args.model_path is None:
            clf.load_pretrained_model()
        else:
            clf.load_pretrained_model(args.model_path)
        _set_target(clf)
        # print("ok step3")
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

        _set_target(clf)
        return
    except Exception as e:
        crash_prompt(f"Can't import plugin {args.target}")
        crash_prompt(f"Error was: {e}")
        return


def _set_target(clf):
    global_state.target = clf
    success_prompt('Target set!')



def train_model(model, path, label):
    print("ok inside train 1")
    try:
        print("global_state.data_paths ", global_state.data_paths)
        for file_path in path:
            with open(file_path, 'rb') as handle:
                bytecode = handle.read()
            print("ok inside train 2")
            print("global_state.target ", global_state.target)
            net: CClassifierEnd2EndMalware = global_state.target
            x = End2EndModel.bytes_to_numpy(bytecode, net.get_input_max_length(), net.get_embedding_value(),
                                            net.get_is_shifting_values())
            x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x)
            model.train(True)
            criterion = nn.BCELoss()
            optimizer = Adam(model.parameters(), lr=0.01)
            scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5,
                                          threshold=0.001, min_lr=0.00001, mode='max')
            print("ok inside train 3")
            epochs = 3
            y_pred = 0
            for epoch in range(epochs):
                print("ok inside train 4 ", model.embed(x))
                y_pred = model.embedd_and_forward(model.embed(x))
                print("ok inside train 5")
                # y_pred = y_pred
                label = torch.tensor(label)  # Convert label to a PyTorch tensor

                # Reshape label to match the shape of y_pred
                label = label.view(y_pred.shape)
                print("ok inside train 6 ", y_pred)
                print("ok inside train 7 ", label)
                loss = criterion(y_pred, label)
                print("ok inside train 8 ", loss)
                optimizer.zero_grad()
                print("ok inside train 9")
                loss.backward()
                print("ok inside train 10")
                optimizer.step()
                print("ok inside train 11")
            model.eval()
            print("ok inside train 12")
            f1 = f1_score(label.detach().numpy(), y_pred.detach().numpy())
            scheduler.step(f1)
            print("ok inside train 13")
    except Exception as e:
        print(e)
    return model
