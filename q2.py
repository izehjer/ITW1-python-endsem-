#Author: Mohmmad Anas Khan 
#Roll no: 20075054
#python end sem q2

#importing all the necesaary librries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearns import preprocessing 
from sklearns import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#loading the file and dividing into training and test data set 
filename= 'Problem2.txt'
data= pd.read_csv( filename , delimiter='\t' )
data_train , data_test = train_test_split(data , test_size=0.2)

#extarcting features and labels from training data set 
#scaling each features in (0,1) range 
y_train = np.array(data_train['Outcome'])
x_train=np.array(data_train[data_train.columns[:8]])
scale = preprocessing.MinMaxScaler.fit(x_train)
x_train_scaled = scale.fit_transform(x_train)

#extarcting features and labels from test data set 
#scaling each features in (0,1) range 
y_test = np.array(data_test['Outcome'])
x_test=np.array(data_test[data_test.columns[:8]])
x_test_scaled = scale.fit_transform(x_test)

#declaring the neural network model and training the model 
nn = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,6), 
                    random_state=1,
                    max_iter=1000)
                    
nn.fit(x_train_scaled, y_train)
# making predictions using the trained model
# finding the accuracy of the predictions and printing it

y_train_pred = nn.predict(x_train_scaled)
y_test_pred = nn.predict(x_test_scaled)
accuracy_train = (accuracy_score(y_train_pred, y_train))*100
accuracy_test = (accuracy_score(y_test_pred, y_test))*100
print("Accuracy on training data: {}%".format(accuracy_train))
print("Accuracy on test data: {}%".format(accuracy_test))

# plotting and visualizing the data

fig, ax = plt.subplots(2,2)

pos = np.squeeze(np.where(y_train == 1))
neg = np.squeeze(np.where(y_train == 0)) 

x = np.array(data_train['Age'])
y = np.array(data_train['DiabetesPedigreeFunction'])
z = np.array(data_train['Glucose'])


# histogram of diabetic and non-diabetic people distributed 
# on the basis of age, from training dataset 

ax[0,0].hist([x[pos], x[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[0,0].set_title('Distribution by Age')
ax[0,0].set_xlabel('Age')
ax[0,0].set_ylabel('Frequency')
ax[0,0].legend()


# histogram of diabetic and non-diabetic people distributed 
# on the basis of DPF, from training dataset 

ax[0,1].hist([y[pos], y[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[0,1].set_title('Distribution by DPF')
ax[0,1].set_xlabel('Diabetes Pedigree Function (DPF)')
ax[0,1].set_ylabel('Frequency')
ax[0,1].legend()


# histogram of diabetic and non-diabetic people distributed 
# on the basis of Glucose, from training dataset 

ax[1,0].hist([z[pos], z[neg]], bins=10, label=['Diabetic', 'Not Diabetic'])
ax[1,0].set_title('Distribution by Glucose')
ax[1,0].set_xlabel('Glucose')
ax[1,0].set_ylabel('Frequency')
ax[1,0].legend()


# scatter plot of predictions 

correct_pred = np.where(y_test == y_test_pred)
wrong_pred = np.where(y_test != y_test_pred)
ax[1,1].scatter(X_test[correct_pred, 6], X_test[correct_pred, 7], c='r', label='Correct Prediction')
ax[1,1].scatter(X_test[wrong_pred, 6], X_test[wrong_pred, 7], c='k', label='Wrong Prediction')
ax[1,1].set_title('Predictions')
ax[1,1].set_xlabel('DPF')
ax[1,1].set_ylabel('Age')
ax[1,1].legend()

plt.show()



