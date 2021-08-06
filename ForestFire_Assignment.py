# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 04:01:25 2021

@author: ASUS
"""

#Importing all the Necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Loading the dataset
file=pd.read_csv(r"D:\Data Science Assignments\Python-Assignment\SVM\forestfires.csv")


#Normalization function
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x

#Normalizing the data and defining the predictor and target columns 
x=norm(file.iloc[:,2:30])
y=file.loc[:,'size_category']
y.unique()

#Splitting dataframe into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#SVM Model, kernel = Linear
model1=SVC(kernel='linear')
model1.fit(x_train,y_train)
pred1=model1.predict(x_test)#Predicting the values
#Checking accuracy of the model
np.mean(y_test==pred1)
pd.crosstab(y_test,pred1)

#SVM Model, kernel = Radial
model2=SVC(kernel='rbf')
model2.fit(x_train,y_train)
pred2=model2.predict(x_test)#Predicting the values
#Checking accuracy of the model
np.mean(y_test==pred2)
pd.crosstab(y_test,pred2)

#SVM Model, kernel = Polynomial
model3=SVC(kernel='poly',degree=3)
model3.fit(x_train,y_train)
pred3=model3.predict(x_test)#Predicting the values
#Checking accuracy of the model
np.mean(y_test==pred3)
pd.crosstab(y_test,pred3)
