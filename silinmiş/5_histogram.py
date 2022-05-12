# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kütüphane
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#2.veri yükleme
veriler=pd.read_csv("veriler.csv")
everiler=pd.read_csv("everiler.csv")

#veri önişleme- nan değerlere ortalama yaztırma
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy="mean")
boy=veriler[['boy']]
boykilo=veriler[["boy","kilo"]]
yas=everiler.iloc[:,1:4].values
#fit_transform
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print (yas)

#encoding 
ulke=veriler.iloc[:,0:1].values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke [:,0]=le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print (ulke)
print (yas)

#dataframe dönüştürme

sonuc= pd.DataFrame(data=ulke, index =range(22), columns =["fr","tr","us"])
print (sonuc)
sonuc2=pd.DataFrame(data=yas, index=range(22), columns= ["boy", "kilo", "yas"])
print (sonuc2)
cinsiyet=veriler.iloc[:,-1].values
print (cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet, index= range (22), columns=["cinsiyet"])
print (sonuc3)

#datafremelere birleştirme
s=pd.concat([sonuc,sonuc2], axis=1)
print (s)
s2=pd.concat([s,sonuc3],axis=1)
print (s2)


# verilerin bölünmesi, standartlaştırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#veri benzeştirme The standard score of a sample x is calculated as:
# z = (x - u) / s
# where u is the mean of the training samples or zero 
# and s is the standard deviation of the training samples or one .

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#histogram çizdirme 

import seaborn as sns
den=[1,1,1,2,3,3,4,4,4,4,4,4,5,5,6,6,6,6,6,6,6,7,7,7,7,8,8,]
print (boy)
print (len(boy))
sns.set()
his= plt.hist(den,bins=4)

plt.show()

#ECDF 



