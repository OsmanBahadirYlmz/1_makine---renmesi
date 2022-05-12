# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:01:12 2020

@author: oby_pc
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression 

df=pd.read_csv("titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model.fit(X, y)
y_pred=model.predict(X)
#K-FOLD The k is the number of chunks we split our dataset into.
# The standard number is 5
#kf normal split işlemi yapıyor ama 3 defa,çoklu tren

from sklearn.model_selection import KFold
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]


kf = KFold(n_splits=3, shuffle=True)
chunks=kf.split(X) #chunks generator-dont need
print(list(kf.split(X)))
#-------------------------
splits = list(kf.split(X))
first_split = splits[0]
train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#her bir splitti loopla test edelim
import numpy as np

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

scores = []
kf = KFold(n_splits=5, shuffle=True)
for a, test_index in kf.split(X):
    X_train, X_test = X[a], X[test_index]
    y_train, y_test = y[a], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))  
print("----------")
print(scores)
print(np.mean(scores))
final_model = LogisticRegression()
final_model.fit(X, y)

from sklearn.model_selection import cross_val_score
''' 
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı

'''
basari = cross_val_score(estimator = model, X=X_train, y=y_train , cv = 4)
print(basari.mean())
print(basari.std())
