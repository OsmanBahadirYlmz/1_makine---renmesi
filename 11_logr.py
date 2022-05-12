# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

veriler = pd.read_csv('veriler.csv')
x = veriler.iloc[:,1:4]
c = veriler.iloc[:,4:]
X = x.values
c = c.values

#encoder: Kategorik -> 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

y = le.fit_transform(c)

# y=np.invert(y)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler

logr=LogisticRegression()
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression()
logr.fit(X_train,y_train)
y_predsc=logr.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
cm = confusion_matrix(y_test,y_predsc)
print(cm)
y_pred2=[0,1,1,1,0,1,1,1]

cm = confusion_matrix(y_test,y_pred2)
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print ("dtc ")
print(cm)



tc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print ("dtc ")
print(cm)






















