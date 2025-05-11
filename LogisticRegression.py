# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:40:34 2025

@author: omer_
"""
from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
df = pd.DataFrame(data=heart_disease.data.features )
df["target"]= heart_disease.data.targets 

print("******************",df.isnull().sum())

df.dropna(inplace=True)   

x=df.drop("target",axis=1)
y=df["target"]
 
print(df.isnull().sum())






x_train, x_test, y_train, y_test=train_test_split(x, y,test_size=0.1,random_state=21)

lr=LogisticRegression(max_iter=1000)

lr.fit(x_train, y_train)


accuracy=lr.score(x_test, y_test)
print("Score: ",accuracy)