#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
import numpy as np
import random
from pathlib import Path
from PIL import Image
fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.util.datasets import load_nursery, load_mnist, load_adult, load_higgs, load_airline
from research.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
    STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--dataset", "-d", choices=FL_DATASETS,
                   help=DATASET_DESC, required=True)
    p.add_argument("--data_path", "-p", help=PATH_DESC,
                   default=os.path.join("examples", "data"))
    p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
                   nargs="+", type=int, required=True)
    p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    return p


def print_statistics(i, x_test_pi, x_train_pi, nb_labels, y_train_pi):
    print('Party_', i)
    print('nb_x_train: ', np.shape(x_train_pi),
          'nb_x_test: ', np.shape(x_test_pi))
    for l in range(nb_labels):
        print('* Label ', l, ' samples: ', (y_train_pi == l).sum())


def readFile(path):
    if str(path).split("/")[-2]=="malignant":
        label=1
    else:
        label=0
    return np.array(Image.open(path)),label


def load_data(normalize=False,data_dir="research/source_data"):
    base=Path(data_dir)
    train=base/"train"
    test=base/"test"
    random.seed(42)

    #Load Train data
    train_files=[x for x in (train/"benign").iterdir()]+[x for x in (train/"malignant").iterdir()]
    train_files=train_files[:200]
    random.shuffle(train_files)
    x_train=np.zeros((len(train_files),224,224,3))
    y_train=np.zeros((len(train_files),1),dtype=int)
    for idx,x in enumerate(train_files):
        x_train[idx],y_train[idx]=readFile(x)
    if(normalize):
        x_train=(x_train-x_train.mean())/x_train.std()

    #Load Test data
    test_files=[x for x in (test/"benign").iterdir()]+[x for x in (test/"malignant").iterdir()]
    test_files=test_files[:100]
    random.shuffle(test_files)
    x_test=np.zeros((len(test_files),224,224,3),dtype=int)
    y_test=np.zeros((len(test_files),1),dtype=int)
    for idx,x in enumerate(test_files):
        x_test[idx],y_test[idx]=readFile(x)
    if(normalize):
        x_test=(x_test-x_test.mean())/x_test.std()
    return (x_train, y_train), (x_test, y_test)

def save_data(nb_dp_per_party, party_folder, label_probs=None,resampling=False):
    """
    Saves MNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = int(np.shape(y_train)[0])
    print(f"{num_train}type{type(num_train)}")
    num_test = np.shape(y_test)[0]
    num_labels = 2
    nb_parties = len(nb_dp_per_party)

    if label_probs:
        # Sample by provided probablities
        train_probs = {0:label_probs[0],1:label_probs[1]}
        test_probs = {0:label_probs[0],1:label_probs[1]}
    else:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[int(label)] / float(num_train) for label in labels}
        test_probs = {label: test_counts[int(label)] /
                      float(num_test) for label in te_labels}


    for idx, dp in enumerate(nb_dp_per_party):
        print(idx,dp)
        train_p = np.array([train_probs[int(y_train[x])]
                            for x in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[int(y_test[idx])] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()

    # Collect arguments
    num_parties = args.num_parties
    dataset = args.dataset
    data_path = args.data_path
    points_per_party = args.points_per_party
    stratify = args.stratify
    create_new = args.create_new
    exp_name = args.name

    # Check for errors
    if len(points_per_party) == 1: # If only one points per party then all parties get same number of data points
        points_per_party = [points_per_party[0] for _ in range(num_parties)]
    elif len(points_per_party) != num_parties:
        parser.error(PER_PARTY_ERR)

    # Create folder to save party data
    folder = os.path.join("research", "data")
    strat = 'balanced' if stratify else 'random'#i.e non IID

    if create_new:
        folder = os.path.join(folder, exp_name if exp_name else str(
            int(time.time())) + '_' + strat)
    else:
        folder = os.path.join(folder, dataset, strat)

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # clear folder of old data
        for f_name in os.listdir(folder):
            f_path = os.path.join(folder, f_name)
            if os.path.isfile(f_path):
                os.unlink(f_path)

    if dataset=="cancer":
        save_data(points_per_party, folder)
