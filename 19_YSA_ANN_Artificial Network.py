# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 00:40:10 2020

@author: oby_pc
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values
#veri on isleme
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]
"""
# yukarıdaki dersin kodu. ülkeleri bölerken 3 kolon yerine
# 2 kolan kullanmış dolayısıyla 2lik koddaki gibi ülkeler ayrılmış
# aşağıdaki kodda benim yazdığım uzun ama daha doğru kod var
# şablon olarak kullanışabilir. 

input dim=12 olacak

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
geo = ohe.fit_transform(veriler.iloc[:,4:5].values).toarray()
geo = pd.DataFrame(data=geo , index=range(geo.shape[0]), columns=["fr","ge","sp"])
gender=le.fit_transform(veriler.iloc[:,5:6])
gender=pd.DataFrame(data=gender , index=range(gender.shape[0]), columns=["male"])
X = pd.concat([veriler.iloc[:,3:4],geo,gender,veriler.iloc[:,6:-1]], axis=1 )

"""

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
"""
ilk add giriş katmanı 
 1 adet çıkış 13 giriş var ortası 6 olduğu için 6 gizli katman kullanıldı. kural değil sanat
 ikinci add 2. gizli katman
 
 tim katmanların initilaze edilmesi lazıom

 üçüncü add. çıkış katmanı 
 gizli katmanlarda lineear çıkışlarda sigmoid kullanılır

optimizer önemli, SGD var adam var. 
optimizer, veriler nasıl değiştirelecek

loss: tahmin etmek istediğimiz değerler 1-0 olduğu için binary croosentropy 
        boşluklar varsa sparse entropy kullanılabilir
        
metrics :        


"""
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6,kernel_initializer='random_uniform', activation = 'relu' , input_dim = 11))
classifier.add(Dense(6,kernel_initializer='random_uniform', activation = 'relu' ))
classifier.add(Dense(1,kernel_initializer='random_uniform', activation = 'sigmoid' ))
#loss küçüldükçe tahmin iyileşir.
classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
#epochs kaç defa döneceği
classifier.fit(X_train, y_train, epochs=50)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)








