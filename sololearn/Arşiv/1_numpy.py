# -*- coding: utf-8 -*-
import pandas as pd
df=pd.read_csv("titanic.csv")
arr=df[["Pclass","Fare","Age"]].values
mask=arr[:,2]<18

# take first 10 values for simplicity
# arr = df[['Pclass', 'Fare', 'Age']].values[:10]

# yaşı 18 den küçük olanları yazdırma,
# ikiside aynı- mask true false,

print(arr[mask]) 
print(arr[arr[:, 2] < 18])
# print(arr.shape) kaça kaçlık bir array

# saydırma

print(mask.sum())
print((arr[:, 2] < 18).sum())

# grafikler

import matplotlib.pyplot as plt

plt.scatter(df["Age"], df["Fare"], c=df['Pclass'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.plot([0, 80], [85, 5])
plt.colorbar()
plt.show()
