# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 01:32:06 2020

@author: oby_pc
"""

#1.kütüphane
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

#2.veri yükleme
veriler=pd.read_csv("maaslar.csv")
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:3]
X=x.values
Y=y.values

#lineer reg
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(x, y, color="red" )
plt.plot (x,lin_reg.predict(x), color = "blue")
plt.show()

#polynomial regression 2. dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X, Y, color="red" )
plt.plot (X,lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

#polynomial regression 4. dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,Y)

plt.scatter(X, Y, color="red" )
plt.plot (X,lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))



#ölçekleme

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
X_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
Y_olcekli=sc2.fit_transform(Y.reshape(-1,1))

from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg.fit(X_olcekli,Y_olcekli)

plt.scatter(X_olcekli,Y_olcekli, color="red")
plt.plot(X_olcekli, svr_reg.predict(X_olcekli), color="blue")

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))









