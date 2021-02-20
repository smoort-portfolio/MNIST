"""
from numba import jit, cuda 
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer 


# normal function to run on cpu
def func(a):								 
	for i in range(10000000): 
		a[i]+= 1	

# function optimized to run on gpu 
#@jit						 
#@jit(nopython=True)
@cuda.jit
def func2(a): 
	for i in range(10000000): 
		a[i]+= 1



n = 10000000							
a = np.ones(n, dtype = np.float64) 
b = np.ones(n, dtype = np.float32) 


start = timer() 
#func(a) 
print("without GPU:", timer()-start)	 

threadsperblock = 32
blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock

start = timer() 
func2[blockspergrid, threadsperblock](a) 
print("with GPU:", timer()-start) 

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
from miniflow_jit import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import datetime

#Read mnist files from current folder .

mndata = MNIST('./')
mndata.gz = True


#train_images, train_labels = mndata.load_training()
#X_ = np.array(train_images).astype(float)
#y_ = np.array(train_labels).astype(float)


test_images, test_labels = mndata.load_testing()
test_X_ = np.array(test_images).astype(float)
test_y_ = np.array(test_labels).astype(float)
y_ = test_y_

scaler = preprocessing.MinMaxScaler()
scaler.fit(test_X_)
#X_ = scaler.transform(X_)
X_ = scaler.transform(test_X_)

saved_trainables = pickle.load(open('parm_files\\trained_parms_h300_e1000_b5.dat', 'rb'))

W1_ = saved_trainables[0].value
b1_ = saved_trainables[1].value
W2_ = saved_trainables[2].value
b2_ = saved_trainables[3].value


# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}
graph = topological_sort(feed_dict)

predict(graph)

rounded_predictions = [round(x[0]) for x in l2.value]
rounded_predictions = np.array(rounded_predictions)
rounded_predictions[rounded_predictions < 0] = 0
rounded_predictions[rounded_predictions > 9] = 9

prediction_accuracy = accuracy_score(y_, rounded_predictions)
print("Prediction accuracy = ", "{:.2%}".format(prediction_accuracy))

d = {"Actual" : y_, "Predicted" : rounded_predictions}
df = pd.DataFrame(d)
#print(df.head)
#Get the confusion matrix
#cf_matrix = confusion_matrix(test_y_, rounded_predictions)
cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(cf_matrix)
cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True, normalize="index")
#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%')
#plt.show()