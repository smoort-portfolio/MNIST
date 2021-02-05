"""
Applying my hand crafted nn to MNIST dataset

"""

print("mnist_nn_classifier -- Start execution")

import numpy as np
from sklearn.utils import shuffle, resample
from miniflow import *
from mnist import MNIST
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time

#Read mnist files from current folder .

mndata = MNIST('./')
mndata.gz = True

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

X_ = np.array(train_images).astype(float)
y_ = np.array(train_labels).astype(float)
test_X_ = np.array(test_images).astype(float)
test_y_ = np.array(test_labels).astype(float)

#print("X_ type", type(X_))
#print("X_ shape", X_.shape)
#print("X_ size", X_.size)
#print(X_[1])

#print("y_ type", type(y_))
#print("y_ shape", y_.shape)
#print("y_ size", y_.size)
#print(y_[1])


# Standardize data
#X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
#X_ = preprocessing.scale(X_)
#X_ = preprocessing.normalize(X_)
scaler = preprocessing.MinMaxScaler(feature_range=[-1, 1])
scaler.fit(X_)
X_ = scaler.transform(X_)
#print(X_[1])


n_features = X_.shape[1]
n_hidden = 25
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}


epochs = 100
# Total number of examples
m = X_.shape[0]
#print("m = ", m)
batch_size = 5
steps_per_epoch = m // batch_size
#print("steps_per_epoch = ", steps_per_epoch)
graph = topological_sort(feed_dict)
#print(graph)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

loss_list = []
loss_drop_list = [0]
print("Starting epochs @", time.time())
start_time = time.time()

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
    if i > 0:
        loss_drop_list.append(loss_list[-1] - (loss/steps_per_epoch))
    loss_list.append(loss/steps_per_epoch)    

end_time = time.time()
print("Ending epochs @", end_time)
print("--- %s seconds ---" % (end_time - start_time))

loss_drop_list[0] = loss_drop_list[1]
df = pd.DataFrame(data={"Loss": loss_list, "Loss Drop": loss_drop_list})
print(loss_list)
print(loss_drop_list)
sns.lineplot(data=df)
plt.show()

print("mnist_nn_classifier -- End execution")