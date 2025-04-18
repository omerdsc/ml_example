# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:44:49 2025

@author: omer_
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

iris=load_iris()

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x, y,train_size=0.8,random_state=42)

#DT model oluştur ve train et

tree_clf=DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42) #♣criterion="entropy"

tree_clf.fit(x_train,y_train)

y_pred=tree_clf.predict(x_test)

accuracy=accuracy_score(y_test, y_pred)
print("DesicionTree score",accuracy )

cm=confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(15,10))
plot_tree(tree_clf,filled=True,feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()

feature_importances=tree_clf.feature_importances_
feature_names=iris.feature_names
feature_importances_Sorted=sorted(zip(feature_importances,feature_names),reverse=True)

for importance ,feature_name in feature_importances_Sorted:
    print(f"{feature_name}: {importance}")

#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris=load_iris()

n_classes=len(iris.target_names)
plot_colors= "ryb"

for pairidx, pair  in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    x=iris.data[:,pair]
    y=iris.target
    
    clf=DecisionTreeClassifier().fit(x,y)
    ax=plt.subplot(2,3,pairidx+1)
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.4)
    DecisionBoundaryDisplay.from_estimator(clf,x,
                                             cmap=plt.cm.RdYlBu,
                                             response_method="predict",
                                             ax=ax,
                                             xlabel=iris.feature_names[pair[0]],
                                             ylabel=iris.feature_names[pair[1]])
    for i, color in zip(range(n_classes),plot_colors):
        idx=np.where(y==i)
        plt.scatter(x[idx,0], x[idx,1],c=color,label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu,
                    edgecolors="black")
    
plt.legend()

#%%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,root_mean_squared_error


diabets=load_diabetes()

x=diabets.data
y=diabets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

reg=DecisionTreeRegressor(random_state=42)
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

mse=mean_squared_error(y_test, y_pred)
print("mse score: ",mse)

rmse=root_mean_squared_error(y_test, y_pred)
print("rmse score",rmse)