import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mp

cancer=load_breast_cancer()
df = pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target
x= cancer.data
y=cancer.target

knn = KNeighborsClassifier(n_neighbors=3)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)#veri setimizi train ve test olmak üzere 2 ye böldük

knn.fit(x_train, y_train)#knn algoritması çalıştı

acc_list=[]
k_list=[]
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)#knn algoritması çalıştı

    y_predict=knn.predict(x_test)
    accuracy=accuracy_score(y_test,y_predict)
    acc_list.append(accuracy)
    k_list.append(k)
    print("y degeri:",accuracy)
    
mp.figure()
mp.plot(k_list,acc_list,linestyle="-",marker="o")
mp.xticks(k_list)
mp.grid(True)