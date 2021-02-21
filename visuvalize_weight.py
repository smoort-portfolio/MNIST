import pickle
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn import preprocessing


label = 0

# load the saved weights & biases from disk
saved_trainables = pickle.load(open("parm_files\\ln_trained_parms_e100_b5.dat", 'rb'))

W = saved_trainables[0]
b = saved_trainables[1]

"""
mndata = MNIST('./')
mndata.gz = True
test_images, test_labels = mndata.load_testing()
test_X_ = np.array(test_images).astype(float)
test_y_ = np.array(test_labels).astype(float)


test_X_ = pickle.load(open("parm_files\\new_X_.mnist", 'rb'))
test_y_ = pickle.load(open("parm_files\\new_y_.mnist", 'rb'))


print("label = " + str(test_y_[label]))

#X = np.full(784, 255.0)
X = test_X_[label]

WT = np.transpose(W)
XT = np.transpose(X)
#print(X.shape)
#print(W.shape)
#print(b.shape)
#print(WT.shape)
#print(W[0].shape)
#print(b[0].shape)

smin = 0
smax = 255


fig=plt.figure(figsize=(2, 5))
for i in range(10):
    w = WT[i]
    new_X = X * w    
    max = np.max(new_X)
    min = np.min(new_X)
    X_scaled = (new_X - min) * (smax - smin) / (max - min) + smin
    #img = X_scaled.reshape((28, 28))
    img = new_X.reshape((28, 28))
    fig.add_subplot(2, 5, i+1)
    plt.imshow(img, cmap="gray")
    label = "Y = " + str(i)
    plt.title(label, fontsize=8)

"""

WT = np.transpose(W)
print(W.shape)
print(WT.shape)
fig=plt.figure(figsize=(2, 5))
smin = 0
smax = 255
for i in range(10):
    w = WT[i]
    max = np.max(w)
    min = np.min(w)
    w = (w - min) * (smax - smin) / (max - min) + smin
    w = w.reshape((28, 28))
    fig.add_subplot(2, 5, i+1)
    plt.imshow(w, cmap="gray")
    label = "Y = " + str(i)
    plt.title(label, fontsize=8)

plt.show()