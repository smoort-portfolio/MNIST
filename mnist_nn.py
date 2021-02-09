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
import pickle
from sklearn.metrics import accuracy_score
import datetime
from sklearn.metrics import confusion_matrix

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
#scaler = preprocessing.MinMaxScaler(feature_range=[-1, 1])
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_)
X_ = scaler.transform(X_)
#print(X_[1])


n_features = X_.shape[1]
n_hidden = 50
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


epochs = 350 # best value = 1000
# Total number of examples
m = X_.shape[0]
#print("m = ", m)
batch_size = 5 # best value = 5
steps_per_epoch = m // batch_size
#print("steps_per_epoch = ", steps_per_epoch)
graph = topological_sort(feed_dict)
#print(graph)

parm_filename = "parm_files\\trained_parms" + "_h" + str(n_hidden) + "_e" + str(epochs) + "_b" + str(batch_size) + ".dat"

trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

loss_list = []
loss_drop_list = [0]
print("Starting epochs @", datetime.datetime.now())
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
print("Ending epochs @", datetime.datetime.now())
print("--- %s seconds ---" % (end_time - start_time))

# save the trained weights & biases to disk
#parm_filename = "parm_files\\trained_parms" + "_h" + str(n_hidden) + "_e" + str(epochs) + "_b" + str(batch_size) + ".dat"
pickle.dump(trainables, open(parm_filename, 'wb'))

# visualize training loss
loss_drop_list[0] = loss_drop_list[1]
df = pd.DataFrame(data={"Loss": loss_list, "Loss Drop": loss_drop_list})
#print(loss_list)
#print(loss_drop_list)
#sns.lineplot(data=df)
#plt.show()

"""

Prediction section starts here

"""
# load the saved weights & biases from disk
saved_trainables = pickle.load(open(parm_filename, 'rb'))

W1 = saved_trainables[0]
b1 = saved_trainables[1]
W2 = saved_trainables[2]
b2 = saved_trainables[3]

# Remove the cost function layer (last layer) of the trained model as we will need predictions now
graph.pop(-1)

# Scale the test inputs to the same level as the training inputs
X.value = scaler.transform(test_X_)

# Run the preduction by feeding the test input to the trained model
predict(graph)

rounded_predictions = [round(x[0]) for x in l2.value]
print(rounded_predictions[:5])

prediction_file = "parm_files\\predictions" + "_h" + str(n_hidden) + "_e" + str(epochs) + "_b" + str(batch_size) + ".dat"
pickle.dump(rounded_predictions, open(prediction_file, 'wb'))

# Calculate accuracy of predicts against test labels
"""
first_image = test_X_[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
"""

prediction_accuracy = accuracy_score(test_y_, rounded_predictions)
print("Prediction accuracy = ", "{:.2%}".format(prediction_accuracy))

# Append accuracy information at the top of file along with training parameters
accuracy_history = {
    "Accuracy": "{:.2%}".format(prediction_accuracy),
    "Execution time": "%s secs" % (end_time - start_time),
    "hidden_layers": n_hidden,
    "epochs": epochs,
    "batch_size": batch_size
}
accuracy_filename = "parm_files\\accuracy_history.txt"

with open(accuracy_filename,'r') as f:
      existing_data = f.read()
with open(accuracy_filename,'w') as f:
      f.write(str(datetime.datetime.now()) + " " + str(accuracy_history) + "\n")
with open(accuracy_filename,'a') as f:
      f.write(existing_data)

#Get the confusion matrix
d = {"Actual" : test_y_, "Predicted" : rounded_predictions}
df = pd.DataFrame(d)

cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%')
#sns.heatmap(cf_matrix, annot=True)
plt.show()

print("mnist_nn_classifier -- End execution")