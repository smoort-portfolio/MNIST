"""
Applying my hand crafted linear classifier to MNIST dataset

"""

print("mnist_linear_classifier --- Start execution")

from mnist import MNIST
import numpy as np
import pandas as pd
from sklearn.utils import shuffle, resample
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import pickle
from sm_linear_classifier import *

epochs = 10 # best value = 1000
batch_size = 10 # best value = 5
learning_rate = 1.5

#Read mnist files from current folder .

mndata = MNIST('./')
mndata.gz = True

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

"""
train_images_filename = "parm_files\\mnist_train_images.mnist"
test_images_filename = "parm_files\\mnist_test_images.mnist"
train_labels_filename = "parm_files\\mnist_train_labels.mnist"
test_labels_filename = "parm_files\\mnist_test_labels.mnist"
train_images = pickle.load(open(train_images_filename, 'rb'))
train_labels = pickle.load(open(train_labels_filename, 'rb'))
test_images = pickle.load(open(test_images_filename, 'rb'))
test_labels = pickle.load(open(test_labels_filename, 'rb'))
"""

X_ = np.array(train_images).astype(float)
y_ = np.array(train_labels).astype(float)
test_X_ = np.array(test_images).astype(float)
test_y_ = np.array(test_labels).astype(float)

# Normalize and zero center training and test input data
X_ = (X_ - 128)/255
test_X_ = (test_X_ - 128)/255

# One hot enode lables
lb = preprocessing.LabelBinarizer()
lb.fit([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
y_ = lb.transform(y_)

n_labels = 10
n_features = X_.shape[1]
W = np.random.randn(n_features, n_labels)
b = np.zeros(n_labels)
trainables = [W, b]

# Total number of examples
m = X_.shape[0]
print("Total number of examples = {}".format(m))

#batch_size = 10 # best value = 5
steps_per_epoch = m // batch_size
#print("steps_per_epoch = ", steps_per_epoch)

loss_list = []
loss_drop_list = [0]

print("Starting epochs @", datetime.datetime.now())
start_time = epoch_start_time = time.time()

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Step 2
        mse,  W, b = forward_and_backward(X_batch, y_batch, W, b, learning_rate)

        loss += mse

    #print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
    epoch_time = time.time()
    #print("Time taken = ", epoch_time - epoch_start_time)
    epoch_start_time = epoch_time
    if i > 0:
        loss_drop_list.append(loss_list[-1] - (loss/steps_per_epoch))
        if i%20 == 0:
            predictions = predict(test_X_, W, b)
            predictions = np.array(predictions)
            prediction_accuracy = accuracy_score(test_y_, predictions)
            print("Prediction accuracy = ", "{:.2%}".format(prediction_accuracy))
    loss_list.append(loss/steps_per_epoch)

end_time = time.time()
print("Ending epochs @", datetime.datetime.now())
print("--- %s seconds ---" % (end_time - start_time))

parm_filename = "parm_files\\ln_trained_parms" + "_e" + str(epochs) + "_b" + str(batch_size) + ".dat"
# save the trained weights & biases to disk
pickle.dump(trainables, open(parm_filename, 'wb'))

# visualize training loss
loss_drop_list[0] = loss_drop_list[1]
df = pd.DataFrame(data={"Loss": loss_list, "Loss Drop": loss_drop_list})
#sns.lineplot(data=df)
#plt.show()

"""

#Prediction section starts here

"""
# load the saved weights & biases from disk
saved_trainables = pickle.load(open(parm_filename, 'rb'))

W = saved_trainables[0]
b = saved_trainables[1]


# Run the preduction by feeding the test input to the trained model
predictions = predict(test_X_, W, b)

predictions = np.array(predictions)
#print("Actual = ", test_y_)
#print("Prediction =", predictions)
prediction_accuracy = accuracy_score(test_y_, predictions)
print("Prediction accuracy = ", "{:.2%}".format(prediction_accuracy))

prediction_file = "parm_files\\predictions" + "_e" + str(epochs) + "_b" + str(batch_size) + ".dat"
pickle.dump(predictions, open(prediction_file, 'wb'))

# Append accuracy information at the top of file along with training parameters
accuracy_history = {
    "Model Type": "Linear Classifier",
    "Accuracy": "{:.2%}".format(prediction_accuracy),
    "Execution time": "%s secs" % (end_time - start_time),
    "learning_rate": learning_rate,
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
d = {"Actual" : test_y_, "Predicted" : predictions}
df = pd.DataFrame(d)

cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(cf_matrix)
#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%')
#sns.heatmap(cf_matrix, annot=True)
#plt.show()

print("mnist_linear_classifier -- End execution")