import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
import random
from pathlib import Path
from PIL import Image
fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)
from ibmfl.util.datasets import load_nursery, load_mnist, load_adult, load_higgs, load_airline
from research.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
    STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC, RATIO_PER_PARTY, RATIO_PER_PARTY_ERR


INPUT_SIZE=112

def setup_parser():
    """
    Sets up the parser for Python script
    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
                   nargs="+", type=int, required=True)
    p.add_argument("--ratio_per_party", "-r", help=RATIO_PER_PARTY,
                   nargs="+", type=float, required=False, default=[None])                   
    # p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
    p.add_argument("--name", "-N",help=NAME_DESC,required=True)
    return p


def print_statistics(i, x_test_pi, x_train_pi, nb_labels, y_train_pi):
    print('Party_', i)
    print('nb_x_train: ', np.shape(x_train_pi),
          'nb_x_test: ', np.shape(x_test_pi))
    for l in range(nb_labels):
        print('* ', "Benign" if l else "Malignant", ' samples: ', (y_train_pi == l).sum())


# def readFile(path):
#     if str(path).split("/")[-2]=="malignant":
#         label=1
#     else:
#         label=0
#     return np.array(Image.open(path).resize((112,112))),label

def split_data(images,x,y,splitPercent=0.8):
    idxs=[i for i in range(len(x))]
    random.seed(10)
    random.shuffle(idxs)
    
    splitOn=int(len(idxs)*splitPercent)
    train_idxs=idxs[0:splitOn]
    test_idxs=idxs[splitOn:]
    
    train_im=images[train_idxs]
    train_tab=x[train_idxs]
    train_y=y[train_idxs]

    test_im=images[test_idxs]
    test_tab=x[test_idxs]
    test_y=y[test_idxs]
        
    return (train_im, train_tab, train_y, test_im, test_tab, test_y)


def load_data(normalize=False,data_dir="multiinput/source_data"):
    data=pd.read_pickle("multiinput/source_data/data.pickle")
    df = data.sample(frac=1,random_state=69).reset_index(drop=True)
    df["anatom_general_site"].fillna(value="unknown",inplace=True)
    #dropping irrelevant columns and creating dummy variables
    df=df.drop(["name","image_name"],axis=1)
    dummy_df=pd.get_dummies(df,columns=["sex","anatom_general_site"])
    #getting images
    dummy_df["image"]=dummy_df["image"].map(lambda x: np.asarray(x,dtype="uint8"))
    images=np.array([i[0] for idx,i in enumerate(dummy_df[["image"]].values)])
    images=images/255
    #getting y values
    y=dummy_df["labels"].map(lambda x: 1 if x =="benign" else 0).values
    # y=np.eye(2)[y]
    dummy_df=dummy_df.drop(["image","labels"],axis=1)

    x=dummy_df.to_numpy(dtype=float)
    return split_data(images,x,y)

def save_data(nb_dp_per_party, party_folder, label_probs=None):
    """
    Saves MNIST party data
    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    train_im, train_tab, train_y, test_im, test_tab, test_y = load_data()
    labels, train_counts = np.unique(train_y, return_counts=True)
    te_labels, test_counts = np.unique(test_y, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = int(np.shape(train_y)[0])
    print(f"NUMTRAIN {num_train}")

    print(f"{num_train}type{type(num_train)}")
    num_test = np.shape(test_y)[0]
    num_labels = 2
    nb_parties = len(nb_dp_per_party)

    if label_probs:
        # Sample by provided probablities
        train_probs = [{1:prob,0:1-prob} for prob in label_probs]
        #test_probs = {0:label_probs[0],1:label_probs[1]}
    else:
        # Sample according to source label distribution
        for n in labels:
            print(type(n),n)
            # print(int(n))
        train_probs = [{
            label: train_counts[int(label)] / float(num_train) for label in labels}]*nb_parties
        #test_probs = [{label: test_counts[int(label)] /
        #              float(num_test) for label in te_labels}]*nb_parties

    def update_probs_and_data(probs, x ,images,y, indices):
        probs=np.delete(probs,indices)
        x=np.delete(x,indices,axis=0)
        y=np.delete(y,indices,axis=0)
        images=np.delete(images,indices,axis=0)
        #updating the probabilities to get the sum to 1
        probs=probs/sum(probs)
        return probs,x,images,y
    print(num_train)
    for idx, dp in enumerate(nb_dp_per_party):
        print(idx,dp)
        train_p = np.array([train_probs[idx][int(train_y[x])]
                            for x in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p,replace=False)

        #test_p = np.array([test_probs[int(y_test[idx])] for idx in range(num_test)])
        #test_p /= np.sum(test_p)
        #
        ## Split test evenly
        #test_indices = np.random.choice(
        #    num_test, int(num_test / nb_parties), p=test_p)

        train_im_pi = train_im[train_indices]
        train_tab_pi = train_tab[train_indices]
        train_y_pi = train_y[train_indices]
        test_im_pi = test_im[0:num_test//nb_parties]
        test_tab_pi = test_tab[0:num_test//nb_parties]
        test_y_pi = test_y[0:num_test//nb_parties]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, train_im=train_im_pi,train_tab=train_tab_pi, train_y=train_y_pi,
                 test_im=test_im_pi, test_tab=test_tab_pi, test_y=test_y_pi)

        print_statistics(idx, test_im_pi, train_im_pi, num_labels, train_y_pi)
        
        #deleting data that is already used by
        train_p, train_im,train_tab, train_y=update_probs_and_data(train_p, train_im,train_tab, train_y, train_indices)
        num_train = int(np.shape(train_y)[0])
        num_test = np.shape(test_y)[0]
        
        print('Finished! :) Data saved in ', party_folder)


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()

    # Collect arguments
    num_parties = args.num_parties
    points_per_party = args.points_per_party
    ratio_per_party = args.ratio_per_party
    exp_name = args.name

    # Check for errors
    if len(points_per_party) == 1: # If only one points per party then all parties get same number of data points
        points_per_party = [points_per_party[0] for _ in range(num_parties)]
    elif len(points_per_party) != num_parties:
        parser.error(PER_PARTY_ERR)

    if len(ratio_per_party) == 1: # If only one points per party then all parties get same number of data points
        ratio_per_party = ratio_per_party[0]
        if ratio_per_party: #if it is not none (none is the default value)
            ratio_per_party=[ratio_per_party]*num_parties
    elif len(ratio_per_party) != num_parties:
        parser.error(RATIO_PER_PARTY_ERR)

    # Create folder to save party data
    folder = os.path.join("multiinput", "data")

    folder = os.path.join(folder, exp_name if exp_name else str(
        int(time.time())))

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # clear folder of old data
        for f_name in os.listdir(folder):
            f_path = os.path.join(folder, f_name)
            if os.path.isfile(f_path):
                os.unlink(f_path)

    save_data(points_per_party, folder, ratio_per_party)


