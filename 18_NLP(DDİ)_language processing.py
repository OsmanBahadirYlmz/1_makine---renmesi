# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:34:47 2020

@author: oby_pc
"""

import numpy as np
import pandas as pd
import re
import nltk
nltk.download("stopwords")
yorumlar = pd.read_csv("Restaurant_Reviews.csv")
from nltk.stem.porter import PorterStemmer
# gövdesine ayarmak
ps=PorterStemmer()

from nltk.corpus import stopwords

#Preprocessing
derlem = []
for i in range(yorumlar.shape[0]):
    #başında ^işareti değil demek. bunları bul ve boş karakterle değiştir diyoruz
    yorum = re.sub("[^a-zA-Z]"," ", yorumlar["Review"] [i])
    #tümünü küçük harfle değiştirdik
    yorum=yorum.lower()
    # tüm kelimeleri ayırıp liste haline getirdik
    yorum= yorum.split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum= " ".join(yorum)
    derlem.append(yorum)

#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(derlem).toarray()
y= yorumlar.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print (cm)



