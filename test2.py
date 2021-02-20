import pickle
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

incorrect_predictions_filename = "parm_files\\incorrect_predictions.dat"
incorrect_predictions = pickle.load(open(incorrect_predictions_filename, 'rb'))

mndata = MNIST('./')
mndata.gz = True
#train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
test_X_ = np.array(test_images).astype(float)
#y_ = np.array(train_labels).astype(float)

test_y_ = []
predictions = []
for x in incorrect_predictions:
    test_y_.append(x[1])
    predictions.append(x[2])
    

cf_matrix = confusion_matrix(test_y_, predictions)
print(cf_matrix)


#Get the confusion matrix
d = {"Actual" : test_y_, "Predicted" : predictions}
df = pd.DataFrame(d)

cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(cf_matrix)

#pivot = pd.pivot_table(df, values='Predicted', index=['Actual'], columns=['Predicted'], aggfunc=np.sum)
#pivot = pd.pivot_table(df, values='Predicted', index=['Actual'], columns=['Predicted'], aggfunc='count')
#print(pivot)

cf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True, normalize="index")
#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%')
#plt.show()

fig=plt.figure(figsize=(5, 5))
columns = 10
rows = math.ceil(len(incorrect_predictions)/columns)
#for i in range (len(incorrect_predictions)):
for i in range (200):
    img = test_X_[incorrect_predictions[i][0]]
    img = img.reshape((28, 28))
#    fig.add_subplot(rows, columns, i+1)
    fig.add_subplot(10, 20, i+1)
    plt.imshow(img)
    label = "Y=" + str(incorrect_predictions[i][1]) + "  P=" + str(incorrect_predictions[i][2])
    plt.title(label, fontsize=6)
    plt.axis('off')
plt.show()