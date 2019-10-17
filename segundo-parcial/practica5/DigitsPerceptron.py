'''
11.	Create a new file called DigitsPerceptron.
Use a multilayer perceptron to classify the database “Digits” from “datasets”.
Use solver=”lbfgs” and ten elements with 3 layers hidden_layer_sizes=(10,4).
Split the instances in training and testing set,
train the model and then measure its performance.
'''

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

'''
digits = load_digits()

# activation = relu por default
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 4), activation='relu')
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=1)
clf.fit(X_train, y_train) #entrena
print(clf.score(X_test, y_test)) #testea

# 12. Use “GridSearchCV” to find which is the best configuration using the following params
params = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate_init': [.1, 0.05, 0.01, 0.005, 0.001]
}

grid_search_cv = GridSearchCV(MLPClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.best_estimator_)
y_pred = grid_search_cv.predict(X_test)
print("Accuracy for cancer breast is", accuracy_score(y_test, y_pred) * 100)
'''

'''
MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.01, max_iter=200, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=42, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
'''

'''
13.	Select a Support Vector Machine or a Decision Tree to compare it against a 
multilayer neuronal network. Use the default configuration and the following 
databases:  iris, wine, breast_cancer
'''

from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score

# iris
iris = load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)
# SVM Classifier model
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X_train, y_train)
y_pred_iris = svm_clf.predict(X_test)
result = accuracy_score(y_test, y_pred_iris)

iris = load_iris()
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
result_tree = accuracy_score(y_test, y_pred_tree)

clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred_mlp = clf.predict(X_test)
result_mlp = accuracy_score(y_test, y_pred_mlp)

print("SVM iris:", result)
print('Tree iris: ', result_tree)
print('NN iris: ', result_mlp)
print('\n')

# wine
wine = load_wine()
X = wine.data[:,2:]
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=1)
# SVM Classifier model
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X_train, y_train)
y_pred_wine = svm_clf.predict(X_test)
result = accuracy_score(y_test, y_pred_wine)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
result_tree = accuracy_score(y_test, y_pred_tree)

clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred_mlp = clf.predict(X_test)
result_mlp = accuracy_score(y_test, y_pred_mlp)

print("SVM wine:", result)
print('Tree wine: ', result_tree)
print('NN wine: ', result_mlp)
print('\n')

# beast_cancer
cancer = load_breast_cancer()
X = cancer.data[:, 4:]
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
# SVM Classifier model
svm_clf = SVC(kernel="linear", C=float(200))
svm_clf.fit(X_train, y_train)
y_pred_cancer = svm_clf.predict(X_test)
result = accuracy_score(y_test, y_pred_cancer)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
result_tree = accuracy_score(y_test, y_pred_tree)

clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred_mlp = clf.predict(X_test)
result_mlp = accuracy_score(y_test, y_pred_mlp)

print("SVM breasts:", result)
print('Tree breasts: ', result_tree)
print('NN breasts: ', result_mlp)