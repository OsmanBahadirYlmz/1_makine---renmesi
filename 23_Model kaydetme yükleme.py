# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:53:51 2020

@author: oby_pc
"""

import pandas as pd

url = "http://bilkav.com/satislar.csv"
veriler = pd.read_csv(url)
veriler=veriler.values

X = veriler [:,0:1]
Y= veriler [:,1]
bolme=0.33
from sklearn.model_selection import train_test_split

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=bolme, random_state=0)

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

lr.fit (X_train, Y_train)
print (lr.predict(X_test))


#model kaydetme
import pickle

dosya = "model.kayıt"

pickle.dump(lr,open(dosya,"wb"))

#model okuma

yuklenen = pickle.load (open(dosya,"rb"))
print (yuklenen.predict(X_test))
