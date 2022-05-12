# -*- coding: utf-8 -*-
#kütüphane
import pandas as pd
import numpy as np
import matplotlib as plt

#2.veri yükleme
veriler=pd.read_csv("odev_tenis.csv")

#veri önişleme
hum=veriler.iloc[:,2:3].values
temp=veriler.iloc[:,1:2].values

#encoding Katagorik--> numeric
outlook=veriler.iloc[:,0:1].values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
outlook [:,0]=le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()

windy=veriler.iloc[:,3:4].values
windy [:,-1]=le.fit_transform(veriler.iloc[:,-1])
# windy=ohe.fit_transform(windy).toarray()

play=veriler.iloc[:,4:].values
play [:,0]=le.fit_transform(veriler.iloc[:,4:])
# play=ohe.fit_transform(play).toarray()

#dataframe dönüştürme

sonuc1=pd.DataFrame(data=outlook, index =range(14), columns =["overcast","rainy","sunny"])
sonuc2=pd.DataFrame(data=windy, index=range(14), columns= ["noWind", "Wind"])
sonuc3=pd.DataFrame(data=play, index=range(14), columns= ["noPlay", "play"])
sonuc4=pd.DataFrame(data=temp, index=range(14), columns= ["temp"])
sonuc5=pd.DataFrame(data=hum, index=range(14), columns= ["humadity"])

#pupet elementten kaçma
windy2=pd.DataFrame(data=windy[:,1], index= range (14), columns=["windy"])
play2=pd.DataFrame(data=play[:,1], index= range (14), columns=["play"])


#datafremelere birleştirme

s=pd.concat([sonuc1,sonuc4,sonuc5,windy2,play2], axis=1)

# verilerin bölünmesi,
from sklearn.model_selection import train_test_split


#model hum tahmin
from sklearn.linear_model import LinearRegression

veri=pd.concat([sonuc1,sonuc4,windy2,play2], axis=1)
x_train, x_test, y_train, y_test=train_test_split(veri,hum,test_size=0.33,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)

# backward elemination, 
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int), values=veri, axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(hum,X_l).fit()
print(model.summary())

# p value 0.05 ten büyük olnı eledik
X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(hum,X_l).fit()
print(model.summary())

# # p value 0 dan büyük olanı eledik
# X_l=veri.iloc[:,[0,1,2,3]].values
# X_l=np.array(X_l, dtype=float)
# model = sm.OLS(boy,X_l).fit()
# print(model.summary())

