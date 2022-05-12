# -*- coding: utf-8 -*-

#kütüphane
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

#2.veri yükleme
veriler=pd.read_csv("satislar.csv")
veriler.iloc[19,1]=46970

aylar=veriler[["Aylar"]]
satislar=veriler[["Satislar"]]
satislar2=veriler.iloc[:,:1].values


# verilerin bölünmesi, standartlaştırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=1)

"""
scale etme X_trainden Y_traşni öğrendi, Xtest ten de tahmin oluşturdu.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

#model inşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

tahmin=lr.predict(X_test)
"""
#model inşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

x_train= x_train.sort_index()#saçma değerlere gitmesin diye sıraladık
y_train= y_train.sort_index()
plt.scatter(x_train,y_train,c="blue")
plt.scatter(x_test,tahmin,c="red")
plt.scatter(x_test,y_test,c="black",s=2)


