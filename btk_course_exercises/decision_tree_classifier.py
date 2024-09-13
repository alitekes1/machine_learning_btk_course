from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt

iris = load_iris()

X=iris.data
y=iris.target

#train ve test verisi için main dataset i parçaladık.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

tree_clf=DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42)#model için hiperparametre ayarı yaptık. decision tree de hiperparametre depth tir.

tree_clf.fit(X_train, y_train)# modeli train verisi ile eğitiyoruz.


y_predict=tree_clf.predict(X_test)# test verisi ile predict ediyoruz ve bunun sonucunda predict değerlerini elde ediyoruz.
accu=accuracy_score(y_test, y_predict)#modelin accuracy oranı hesaplıyoruz.
print(accu)

conf_matrix=confusion_matrix(y_test, y_predict)#değerlendirme metriklerinden olan confusion matrix ile modeli değerlendiriyoruz.
print(conf_matrix)

plt.figure(figsize=(15,10))

plot_tree(tree_clf,filled=True,max_depth=5,feature_names=iris.feature_names,class_names=list(iris.target_names))# decision tree yi görselleştirdik