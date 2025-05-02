# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:53:56 2025

@author: omer_
"""

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

oli=fetch_olivetti_faces()

plt.figure()

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i+321],cmap="gray")
    plt.axis("off")
plt.show()

x=oli.data
y=oli.target

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.2,random_state=42)

rf=RandomForestClassifier(n_estimators=100,random_state=42)

rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)

score=rf.score(x_test, y_test)
print(score)


accuracy=accuracy_score(y_test, y_pred)

print(accuracy)

#%%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
california_housing=fetch_california_housing()

x=california_housing.data
y=california_housing.target

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.2,random_state=42)

rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)

score=rf.score(x_test, y_test)
print(score)

mse=mean_squared_error(y_test, y_pred)
print(mse)

rmse=np.sqrt(mse)
print(rmse)
