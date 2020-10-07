import os
import tensorflow as tf
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras

def get_datahandler_config(dh_name, folder_data, party_id, is_agg):
    print(f"Datahandler: {str(folder_data)},str(party")
    data = {
            'name': 'KerasDataHandler',
            'path': 'research.keras.keras_datahandler',
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
        'name': 'FedAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.fedavg_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        # 'name': 'LocalTrainingHandler',
        'name': 'FedAvgLocalTrainingHandler',
        # 'path': 'ibmfl.party.training.local_training_handler'
        'path': 'ibmfl.party.training.fedavg_local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 3,
            'termination_accuracy': 0.9,
            'max_timeout': 60
        },
        'local': {
            'training': {
                'epochs': 3
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
    num_classes = 2
    IMG_SIZE=112
    lr=1e-4
    img_rows, img_cols = IMG_SIZE,IMG_SIZE
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)


    imageInputs=Input(shape=input_shape)
    tabularInputs=Input(shape=(11,))

    convModel=Conv2D(32, (3, 3), activation='relu',kernel_initializer=keras.initializers.glorot_normal())(imageInputs)
    convModel=MaxPooling2D(pool_size=(2, 2))(convModel)
    convModel=Dropout(0.5)(convModel)
    convModel=Flatten()(convModel)
    convModel=Model(inputs=imageInputs,outputs=convModel)

    tabModel=Dense(50,activation='relu',kernel_initializer=keras.initializers.glorot_normal())(tabularInputs)
    tabModel=Model(inputs=tabularInputs,outputs=tabModel)

    # combined=Concatenate()([convModel,tabModel])
    combined=Concatenate()([convModel.output,tabModel.output])

    out=Dense(128, activation='relu',kernel_initializer=keras.initializers.glorot_normal())(combined)
    out=Dropout(0.5)(out)
    out=Dense(num_classes, activation='softmax',kernel_initializer=keras.initializers.glorot_normal())(out)

    model=Model(inputs=[convModel.input,tabModel.input],outputs=out)
    # model=Model(inputs=[imageInputs,tabularInputs],outputs=out)
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
