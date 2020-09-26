"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
from keras.preprocessing.image import ImageDataGenerator
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
        img_rows, img_cols = 112,112
        # if self.file_name is None:
        #     (x_train, y_train), (x_test, y_test) = load_mnist()
        #     # Reduce datapoints to make test faster
        #     x_train = x_train[:nb_points]
        #     y_train = y_train[:nb_points]
        #     x_test = x_test[:nb_points]
        #     y_test = y_test[:nb_points]
        # else:
        try:
            logger.info('Loaded training data from ' + str(self.file_name))
            data_train = np.load(self.file_name)
            x_train = data_train['x_train']
            y_train = data_train['y_train']
            x_test = data_train['x_test']
            y_test = data_train['y_test']
        except Exception:
            raise IOError('Unable to load training data from path '
                          'provided in config file: ' +
                          self.file_name)

        if self.channels_first:
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = (np.eye(num_classes)[y_train]).squeeze()
        y_test = np.eye(num_classes)[y_test].squeeze()
        return (x_train, y_train), (x_test, y_test)
