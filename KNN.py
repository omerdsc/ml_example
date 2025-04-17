# Veri Seti incelenmesi

from sklearn.datasets import load_breast_cancer #göğüs kanseri veri seti
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix #☺doğruluk değeri hesabı
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

cancer=load_breast_cancer()

df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

X=cancer.data
y=cancer.target

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=42)

scaler=StandardScaler() #knn mesafeli bir model olduğundan standatlaştırırız
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

score=knn.score(x_test,y_test)
print("Doğruluk:",score)

accuracy=accuracy_score(y_test, y_pred)

print("Doğruluk:",accuracy)

conf_matrix=confusion_matrix(y_test, y_pred)
print(conf_matrix)


# hipermaremetre ayarlaması
k_values=[]
accuracy_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.figure()
plt.plot(k_values, accuracy_values, marker="o",linestyle="-")
plt.title("k değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)
    
    
