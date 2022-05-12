# -*- coding: utf-8 -*-
#kütüphane
import pandas as pd
import numpy as np
import matplotlib as plt

#2.veri yükleme
veriler=pd.read_csv("odev_tenis.csv")

#veri önişleme

#encoding Katagorik--> numeric
from sklearn import preprocessing

#yeni trick tüm arraya lebel encoding uyguluyor. temp ve hum de çalışmıyor
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

outlook=veriler2.iloc[:,:1]
ohe= preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()

havadurumu=pd.DataFrame(data=outlook , index=range(14), columns=["overcast","rainy","sunny"])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler], axis=1)


# verilerin bölünmesi,
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

#model hum tahmin
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)
print (y_pred)

# backward elemination, 
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int), values=sonveriler, axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler=sonveriler.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int), values=sonveriler, axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

#yeni tahmin nasıl olmus
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)



