"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import logging

import numpy as np
from keras.utils import np_utils

from ibmfl.data.data_handler import DataHandler
from ibmfl.util.datasets import load_mnist

logger = logging.getLogger(__name__)


class KerasDataHandler(DataHandler):
    """
    Data handler for MNIST dataset.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        self.channels_first = channels_first

    def get_data(self, nb_points=500):
        """
        Gets pre-process mnist training and testing data. Because this method
        is for testing it takes as input the number of datapoints, nb_points,
        to be included in the training and testing set.

        :param nb_points: Number of data points to be included in each set
        :type nb_points: `int`
        :return: training data
        :rtype: `tuple`
        """
        num_classes = 2
        IMG_SIZE=112
        img_rows, img_cols = IMG_SIZE,IMG_SIZE

        try:
            logger.info('Loaded training data from ' + str(self.file_name))
            data_train = np.load(self.file_name)
            train_im = data_train['train_im']
            train_tab = data_train['train_tab']
            train_y = data_train['train_y']
            test_im = data_train['test_im']
            test_tab = data_train['test_tab']
            test_y = data_train['test_y']
        except Exception:
            raise IOError('Unable to load training data from path '
                          'provided in config file: ' +
                          self.file_name)


        print('train_im shape:', train_im.shape)
        print(train_im.shape[0], 'train samples')
        print(test_im.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        return ([train_im, train_tab], train_y), ([test_im, test_tab], test_y)


class KerasDataGenerator(DataHandler):

    def __init__(self, data_config):
        super().__init__()

        (train_im, train_y), (test_im, test_y) = load_mnist()
        train_im = train_im.reshape(train_im.shape[0], 28, 28, 1)
        train_im = train_im.astype('float32')
        train_im /= 255
        test_im = test_im.reshape(test_im.shape[0], 28, 28, 1)
        test_im = test_im.astype('float32')
        test_im /= 255

        train_y = np_utils.to_categorical(train_y)
        test_y = np_utils.to_categorical(test_y)
        train_gen = ImageDataGenerator(rotation_range=8,
                                       width_shift_range=0.08,
                                       shear_range=0.3,
                                       height_shift_range=0.08,
                                       zoom_range=0.08)
        test_gen = ImageDataGenerator()

        self.train_datagenerator = train_gen.flow(
            train_im, train_y, batch_size=64)
        self.test_datagenerator = train_gen.flow(test_im, test_y, batch_size=64)

    def get_data(self):

        return self.train_datagenerator, self.test_datagenerator

    def set_batch_size(self, batch_size):
        self.train_datagenerator.set_batch_size(batch_size)
