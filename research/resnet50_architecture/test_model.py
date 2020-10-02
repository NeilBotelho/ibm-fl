import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

data_train = np.load("data/te2_400/data_party0.npz")# Pass a data source of preprocessed images
x_train = data_train['x_train']
y_train = data_train['y_train']
x_test = data_train['x_test']
y_test = data_train['y_test']

num_classes = 2
IMG_SIZE=112
img_rows, img_cols = IMG_SIZE,IMG_SIZE


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    axis=0
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    axis=1
# convert class vectors to binary class matrices
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.transform(y_test.reshape(-1,1)).toarray()

lr=1e-4
num_classes = 2
IMG_SIZE=112
img_rows, img_cols = IMG_SIZE,IMG_SIZE
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

input_shape = (224,224,3)
lr = 1e-5

model = ResNet50(include_top=True,
                 weights= None,
                 input_tensor=None,
                 input_shape=input_shape,
                 pooling='avg',
                 classes=2)

model.compile(optimizer = Adam(lr) ,
              loss = "binary_crossentropy", 
              metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs= 25, batch_size= 64, verbose=2, 
                    
                   )

print(model.evaluate(x_test,y_test))

for label in self.model.metrics_names:
    plt.plot(history.history[label],label=label) 
# plt.plot(history.history["loss"],label="loss")
plt.legend()
plt.savefig(full_path.joinpath("metric_plot"))
