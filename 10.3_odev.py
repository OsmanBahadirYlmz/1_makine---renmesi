# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:05:25 2020

@author: oby_pc
"""
#kütüphaneler
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

#Veriler
veriler=pd.read_csv("Maaslar_yeni.csv")
x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]
X=x.values
Y=y.values
a=veriler["UnvanSeviyesi"].values
print (veriler.corr())

#MLR


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

#p value lere bakalım
from sklearn.linear_model import LinearRegression
lin_reg5 = LinearRegression()
X1=veriler.iloc[:,2:3].values
lin_reg5.fit(X1,Y)

print('------------')
print(r2_score(Y, lin_reg5.predict(X1)))

import statsmodels.api as sn
model=sn.OLS(lin_reg5.predict(X1),X1)
print (model.fit().summary())




#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
print ("polynomial R2 değeri 2.derece")
print(r2_score(Y, lin_reg2.predict(poly_reg2.fit_transform(X))))

poly_reg4 = PolynomialFeatures(degree = 3)
x_poly = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly,y)
print("polynomial R2 değeri 3.derece")
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))


poly_reg3 = PolynomialFeatures(degree = 4)
x_poly = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)
print("polynomial R2 değeri 4.derece")
print(r2_score(Y, lin_reg3.predict(poly_reg3.fit_transform(X))))


#verilerin olceklenmesi SVR
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)


sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

#10 tecrübeli,100 puan almış bir ceo maaşı nedir

ceo=[[10,10,100 ]]
ceo2=[[10,9,83]]
print("lineer reg ceo maaş")
print(lin_reg.predict(ceo))
print("polynomial ceo maaş 2.derece")
print(lin_reg2.predict(poly_reg2.fit_transform(ceo)))
print("polynomial ceo maaş 3.derece")
print(lin_reg4.predict(poly_reg4.fit_transform(ceo)))
print("polynomial ceo maaş 4.derece")
print(lin_reg3.predict(poly_reg3.fit_transform(ceo)))
print("svr ceo maaş ")
print(sc2.inverse_transform(svr_reg.predict(ceo)))
print("DT maaş ")
print(r_dt.predict(ceo))
print("RF maaş ")
print(rf_reg.predict(ceo))

plt.scatter(a,Y,color = 'red')
plt.scatter(a,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()



