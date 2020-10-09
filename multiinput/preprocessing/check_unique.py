import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

party0=np.load("multiinput/data/2_400/data_party0.npz")
party1=np.load("multiinput/data/2_400/data_party1.npz")

x0_train=party0["train_im"]
x1_train=party1["train_im"]

x_train=np.concatenate((x0_train,x1_train),axis=0)
print(x_train.shape)
c_train=np.unique(x_train,return_counts=True,axis=0)

print("train dups: ",sum(c_train[1]>1))


#
## display few images
#w=40
#h=30
#fig1=plt.figure(figsize=(12, 8))
#rows = 4
#columns=4
#for i in range(1, columns*rows+1, 1):
#    ax = fig1.add_subplot(rows, columns, i)
#    if y0_train[-1*i] == 0:
#        ax.title.set_text('Benign')
#    else:
#        ax.title.set_text('Malignant')
#    plt.imshow(x0_train.astype('uint8')[-1*i], interpolation='nearest')
#    plt.axis('off')
#plt.show()
#w=40
#h=30
#fig1=plt.figure(figsize=(12, 8))
#rows = 4
#columns=4
#for i in range(1, columns*rows+1, 1):
#    ax = fig1.add_subplot(rows, columns, i)
#    if y_train[-1*i] == 0:
#        ax.title.set_text('Benign')
#    else:
#        ax.title.set_text('Malignant')
#    plt.imshow(x1_train.astype('uint8')[-1*i], interpolation='nearest')
#    plt.axis('off')
#plt.show()
#
#
