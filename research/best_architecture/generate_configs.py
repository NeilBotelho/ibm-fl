import os
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

def get_datahandler_config(dh_name, folder_data, party_id, is_agg):
    print(f"Datahandler: {str(folder_data)},str(party")
    data = {
            'name': 'KerasDataHandler',
            'path': 'research.best_architecture.keras_datahandler',
            'info': {
                'npz_file': os.path.join(str(folder_data), 'data_party' + str(party_id) + '.npz')
            }
        }
    if is_agg:
            data['info'] = {
                'npz_file': os.path.join("research", "source_data", "train")
            }
    return data

def get_fusion_config():
    fusion = {
        # 'name': 'FedAvgFusionHandler',
        # 'path': 'ibmfl.aggregator.fusion.fedavg_fusion_handler'
        'name': 'IterAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.iter_avg_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'LocalTrainingHandler',
        # 'name': 'FedAvgLocalTrainingHandler',
        'path': 'ibmfl.party.training.local_training_handler'
        # 'path': 'ibmfl.party.training.fedavg_local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 1,
            'termination_accuracy': 0.9,
            'max_timeout': 600
        },
        'local': {
            'training': {
                'epochs': 7
            },
            'optimizer': {
                'lr': 0.01
            }
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    # SUPPORTED_DATASETS = ['mnist']
    # if dataset in SUPPORTED_DATASETS:
    data = get_datahandler_config(
        dataset, folder_data, party_id, is_agg)
    # else:
        # raise Exception(
            # "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None
    lr=1e-3
    num_classes = 2
    IMG_SIZE=112
    img_rows, img_cols = IMG_SIZE,IMG_SIZE
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
        axis=0
    else:
        input_shape = (img_rows, img_cols, 3)
        axis=1
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 kernel_initializer=keras.initializers.glorot_normal()))
    model.add(BatchNormalization(axis=axis,momentum=0.9))
    model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer=keras.initializers.glorot_normal()))
    model.add(BatchNormalization(axis=axis,momentum=0.9))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=keras.initializers.glorot_normal()))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=keras.initializers.glorot_normal()))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=lr,beta_1=0.9),
                  metrics=['accuracy'])


    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, 'compiled_keras.h5')
    model.save(fname)

    K.clear_session()
    # Generate model spec:
    spec = {
        'model_name': 'keras-cnn',
        'model_definition': fname
    }

    model = {
        'name': 'KerasFLModel',
        'path': 'research.models.custom_keras_model',
        'spec': spec
    }

    return model
