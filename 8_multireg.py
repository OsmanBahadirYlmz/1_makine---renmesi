# -*- coding: utf-8 -*-

#kütüphane
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

#2.veri yükleme
veriler=pd.read_csv("veriler.csv")



#veri önişleme- nan değerlere ortalama yaztırma
yas=veriler.iloc[:,1:4].values

#encoding Katagorik--> numeric
ulke=veriler.iloc[:,0:1].values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke [:,0]=le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()


c=veriler.iloc[:,-1:].values
le=preprocessing.LabelEncoder()
c [:,-1]=le.fit_transform(veriler.iloc[:,-1])
ohe= preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()




#dataframe dönüştürme

sonuc= pd.DataFrame(data=ulke, index =range(22), columns =["fr","tr","us"])
print (sonuc)
sonuc2=pd.DataFrame(data=yas, index=range(22), columns= ["boy", "kilo", "yas"])
print (sonuc2)
cinsiyet=veriler.iloc[:,-1].values
print (cinsiyet)

#dummy variable ten kaçmak için sadece 1 element alma
sonuc3=pd.DataFrame(data=c[:,:1], index= range (22), columns=["E"])

#datafremelere birleştirme
s=pd.concat([sonuc,sonuc2], axis=1)
print (s)

s2=pd.concat([s,sonuc3], axis=1)

# verilerin bölünmesi,
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#model cinsiyet tahmin
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)

#model boy tahmin
from sklearn.linear_model import LinearRegression
boy=s2.iloc[:,3:4].values
sol=s2.iloc[:,0:3]
sağ=s2.iloc[:,4:]
veri=pd.concat([sol,sağ], axis=1)
x_train, x_test, y_train, y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train, y_train)
y_pred= regressor.predict(x_test)


# ----------------------------------
# backward elemination, 
import statsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1) #beta değeri belirlemek için veri başına 1 ler ekledik
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# p value 0.05 ten büyük olnı eledik
X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# p value 0 dan büyük olanı eledik
X_l=veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# ------------------------------------




