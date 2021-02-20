import pickle
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

mndata = MNIST('./')
mndata.gz = True
train_images, train_labels = mndata.load_training()
#test_images, test_labels = mndata.load_testing()
X_ = np.array(train_images).astype(float)
y_ = np.array(train_labels).astype(float)


"""
train_images_filename = "parm_files\\mnist_train_images.mnist"
test_images_filename = "parm_files\\mnist_test_images.mnist"
train_labels_filename = "parm_files\\mnist_train_labels.mnist"
test_labels_filename = "parm_files\\mnist_test_labels.mnist"
pickle.dump(train_images[:1], open(train_images_filename, 'wb'))
pickle.dump(test_images[:1], open(test_images_filename, 'wb'))
pickle.dump(train_labels[:1], open(train_labels_filename, 'wb'))
pickle.dump(test_labels[:1], open(test_labels_filename, 'wb'))
"""

new_X_ = []
new_y_ = []
for idx, label in enumerate(y_):
    if label in [6, 8, 9]:
        new_X_.append(X_[idx])
        new_y_.append(y_[idx])
print(len(new_y_))
pickle.dump(new_X_, open("parm_files\\new_X_.mnist", 'wb'))
pickle.dump(new_y_, open("parm_files\\new_y_.mnist", 'wb'))