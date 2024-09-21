# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:35:13 2024

@author: rithu
"""

import tensorflow as tf
print(tf.__version__)

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\rithu\Desktop\Rithuik Python\Deep learning\Churn_Modelling.csv")
X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#creating dummy variable
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)

#concatenating the data frame
X=pd.concat([X,geography,gender],axis=1)

#Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#Initialising the ANNs
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))
#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))
#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
#Adding the output layer
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the ANN to the Training set
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

print(model_history.history.keys())
#summarise history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

