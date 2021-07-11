#author: Mohammad Anas Khan
#Roll no: 20075054
#python - end sem q1 

#imporing all the required directories
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#loading and splitting the data into training and testing data (80-20)

file_name= 'Problem1.txt'
data = pd.read_csv(file_name,delimiter='\t')
data_train , data_test = train_test_split(data , test_size=0.2)

#extracting feature and label vectors from the training data 
#scalaing each features to lie in (0,1) range 
y_train = np.array(data_train['fruit_label'])
x_train = np.array( data_train[data_train.columns[3:]])
scale = preprocessing.MinMaxScaler().fit(x_train)
x_train_scaled = scale.fit_transform(x_train)

#extracting features and label vectors from test data
y_test = np.array(data_test['fruit_label'])
x_test = np.array( data_test[data_test.columns[3:]])
x_test_scaled = scale.fit_transform(x_test)

#declaring the k-nearest neighbour model and training it with trainging data set 
# declaring 5 neighbors with one extra neighbour 
knn = KNeighborsClassifier( n_neighbors=5 )
knn.fit(x_train_scaled, y_train)


# making predictions using the knn model 
# plotting accuracy of the prediction 
knn_y_pred = knn.predict(x_test_scaled)
knn_accuracy = (metrics.accuracy_score(y_test, knn_y_pred))*100

# declaring Logistic Regression model
# training the model with given dataset
 
logisticReg = LogisticRegression(solver='liblinear')
logisticReg.fit(x_train_scaled, y_train)


# making predictions using the trained model
# finding the accuracy of the predictions

log_reg_y_pred = logisticReg.predict(x_test_scaled)
log_reg_accuracy = (metrics.accuracy_score( log_reg_y_pred , y_test))*100

#now comapring and printing the accuracy of both the models 
print("Test Accuracy ( K-Nearest Neighbour )is : {}%".format(knn_accuracy))
print("Test Accuracy ( Logistic Regression )is : {}%".format(log_reg_accuracy))
if knn_accuracy > log_reg_accuracy:
    print("K-Nearest Neighbour algorithm performed better.")
elif knn_accuracy < log_reg_accuracy:
    print("Logistic Regression algorithm performed better.")
else:
    print("Both algorithms performed with same accuracy.")
    
#now plotting and visualizing the data 

fig, ax = plt.subplots(2,2)
fig.set_size_inches(15, 8)


# histogram of fruits type and their frequency in training dataset

ax[0,0].hist(y_train, 4, width=0.5)
label = ['Apple', 'Mandarin', 'Orange', 'Lemon']
rects = ax[0,0].patches
for rect, l in zip(rects, label):
    h = rect.get_height()
    ax[0,0].text((rect.get_x() + rect.get_width()/2), h+0.01, l, ha='center', va='bottom')
ax[0,0].set_title('Fruit Distribution in Training Dataset')
ax[0,0].set_xlabel('Fruits')
ax[0,0].set_ylabel('Frequency')


# scatter plot of fruit distribution according
# to their width and height, from training dataset

groups = data_train.groupby("fruit_name")
for name, group in groups:
    ax[0,1].plot(group["width"], group["height"], marker="o", linestyle="", label=name)
ax[0,1].set_title('Fruit Distribution according to Widht and Height')
ax[0,1].set_xlabel('Width')
ax[0,1].set_ylabel('Height')
ax[0,1].legend()


# scatter plot of fruit predictions using 
# K-Nearest Neighbour algorithm

knn_correct_pred = np.where(y_test == knn_y_pred)
knn_wrong_pred = np.where(y_test != knn_y_pred)
ax[1,0].scatter(x_test[knn_correct_pred, 1], x_test[knn_correct_pred, 2], c='r', label='Correct Prediction')
ax[1,0].scatter(x_test[knn_wrong_pred, 1], x_test[knn_wrong_pred, 2], c='k', label='Wrong Prediction')
ax[1,0].set_title('Predictions by K-Nearest Neighbour Algorithm')
ax[1,0].set_xlabel('Width')
ax[1,0].set_ylabel('Height')
ax[1,0].legend()


# scatter plot of fruit predictions using 
# Logistic Regression Algorithm

lr_correct_pred = np.where(y_test == log_reg_y_pred)
lr_wrong_pred = np.where(y_test != log_reg_y_pred)
ax[1,1].scatter(x_test[lr_correct_pred, 1], x_test[lr_correct_pred, 2], c='r', label='Correct Prediction')
ax[1,1].scatter(x_test[lr_wrong_pred, 1], x_test[lr_wrong_pred, 2], c='k', label='Wrong Prediction')
ax[1,1].set_title('Predictions by Logistic Regression Algorithm')
ax[1,1].set_xlabel('Width')
ax[1,1].set_ylabel('Height')
ax[1,1].legend()

plt.show()






