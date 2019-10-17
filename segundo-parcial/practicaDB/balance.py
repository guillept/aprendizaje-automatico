import numpy as np
from sklearn.model_selection import train_test_split

with open('balance.dat', 'r') as lector:
    lista = lector.readlines()

Xt = [entrada.rstrip('\n').split(',')[:-1] for entrada in lista if not entrada.startswith('@')]
X = []
for elemento in Xt:
    X.append([float(i) for i in elemento])

X=np.asarray(X)
print(X)

yt = [entrada.rstrip('\n').split(',')[-1] for entrada in lista if not entrada.startswith('@')]
y = [float(i) for i in yt]
y=np.asarray(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#maquinas polinomiales
# C = ancho de la barra de soporte. Umbral para ignorar
poly3_kernel_svm_clf=Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, C=5))
])
poly3_kernel_svm_clf.fit(X_train, y_train)

poly10_kernel_svm_clf=Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, C=5))
])
poly10_kernel_svm_clf.fit(X_train, y_train)

#clasificar instancias desconocidas
y_pred1=poly3_kernel_svm_clf.predict(X_test)
y_pred2=poly10_kernel_svm_clf.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(max_depth=10, criterion="entropy")
tree1.fit(X_train, y_train)
y_pred3 = tree1.predict(X_test)

from sklearn.metrics import accuracy_score
result = accuracy_score(y_test, y_pred1)
print("Accuracy MVC 3", result)
result = accuracy_score(y_test, y_pred2)
print("Accuracy MVC 10", result)
result = accuracy_score(y_test, y_pred3)
print("Accuracy arbol", result)

