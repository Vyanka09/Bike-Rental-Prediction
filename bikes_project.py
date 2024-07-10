# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:31:33 2022

@author: Admin
"""

###---- Project for bike rentals prediction -----###

### Step 0 -Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

###Step 1- Read the data 

bikes = pd.read_csv('hour.csv')

###Step 2- Remove some irrevelant columns 

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'],axis=1)
 
##Check null values
bikes_prep.isnull().sum(axis=0)

##Visualize using histogram
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

##Step 3- Data visualization- Continuous
plt.figure()
plt.subplot(2,2,1)
plt.title("Temp vs demand")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')

plt.subplot(2,2,2)
plt.title("Atemp vs demand")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2,c='b')

plt.subplot(2,2,3)
plt.title("Humidity vs demand")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2,c='r')

plt.subplot(2,2,4)
plt.title("Windspeed vs demand")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2,c='c')

##Categorical variables
plt.figure()
plt.subplot(3,3,1)
plt.title("Average demand per season")
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)
##plt.tight_layout()


plt.subplot(3,3,2)
plt.title("Average demand per month")
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)
##plt.tight_layout()


plt.subplot(3,3,3)
plt.title("Average demand per holiday")
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)
##plt.tight_layout()

plt.subplot(3,3,4)
plt.title("Average demand per weekday")
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)

plt.subplot(3,3,5)
plt.title("Average demand per year")
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)

plt.subplot(3,3,6)
plt.title("Average demand per hour")
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)

plt.subplot(3,3,7)
plt.title("Average demand per working day")
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)

plt.subplot(3,3,8)
plt.title("Average demand per weather")
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
colours1 = ['g','b','m','c']
plt.bar(cat_list,cat_average,color=colours1,width=0.5)
plt.tight_layout()

##Check for outliers

bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])

###Step 4- Multiple Linear Regression Assumptions
##1. Check the multicolinearity

confusion = bikes_prep[['temp','atemp','windspeed','humidity','demand']].corr()
print(confusion)
bikes_prep = bikes_prep.drop(['weekday','windspeed','atemp','year','workingday'],axis=1)

##Check autocorrelation for demand
df1 = pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.figure()
plt.acorr(df1,maxlags=12)

###Step 5- Create/modify new features

df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9,bins=20)
plt.figure()
df2.hist(rwidth=0.9,bins=20)

bikes_prep['demand'] = np.log(bikes_prep['demand'])

##Solving problem of autocorrelation
t1 = bikes_prep['demand'].shift(+1).to_frame()
t1.columns = ['t-1']

t2 = bikes_prep['demand'].shift(+2).to_frame()
t2.columns = ['t-2']

t3 = bikes_prep['demand'].shift(+3).to_frame()
t3.columns = ['t-3']

bikes_prep_lag = pd.concat ([bikes_prep,t1,t2,t3],axis=1)
bikes_prep_lag = bikes_prep_lag.dropna()
print(bikes_prep_lag)

###Step 6- Create Dummy Variables and drop first to avoid dummy variables trap
bikes_prep_lag.dtypes
bikes_prep_lag['season']= bikes_prep_lag['season'].astype('category')
bikes_prep_lag['month']= bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']= bikes_prep_lag['hour'].astype('category')
bikes_prep_lag['holiday']= bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']= bikes_prep_lag['weather'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag)

###Step  7- Split the data into train and test

X = bikes_prep_lag.drop(['demand'],axis=1)
Y = bikes_prep_lag [['demand']]

tr_size = 0.7*len(X)
tr_size = int(tr_size)

x_train = X.values[0:tr_size]
x_test = X.values[tr_size:len(X)]

y_train = Y.values[0:tr_size]
y_test = Y.values[tr_size:len(Y)]

### Step 8- Fit and score the model

from sklearn.linear_model import LinearRegression
std_Reg = LinearRegression()
std_Reg.fit(x_train,y_train)

r2_train = std_Reg.score(x_train,y_train)
print("R-square for train "+str(r2_train))
r2_test = std_Reg.score(x_test,y_test)
print("R-square for test "+str(r2_test))

##Create y predictions
Y_predict = std_Reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test , Y_predict ))

print("RMSE= "+str(rmse))