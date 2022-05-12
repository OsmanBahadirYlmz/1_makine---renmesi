# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)
"""
tüm veriler list of list olmalı çünkü veriler biçimli değil
i satırlar
j ler her bir sutun yani ürün

t 7501x20 ye bir list of list
çünkü apyori bunu istiyor

sonuç: 
   basılmış son örneğe bakarsak
    herb and paper almak ground beef alma ihtimalini
    3.29 kat arttırır(lift)
"""
t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))
