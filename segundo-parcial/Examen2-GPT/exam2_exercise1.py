from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

X, y = make_circles(n_samples=200, noise=0.01, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {
    'kernel': ['poly'],
    'C': [1, 10, 100, 1000],
}
# Polynomial Support Vector
grid_search_cv = GridSearchCV(SVC(), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
y_pred = grid_search_cv.predict(X_test)
psv = accuracy_score(y_test, y_pred) * 100


# Decision Tree Classifier
params = {
    'max_leaf_nodes': list(range(2, 100)),
    'criterion': ['entropy']
}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
y_pred = grid_search_cv.predict(X_test)
dtc = accuracy_score(y_test, y_pred) * 100


# Multi-layer Neuronal Network
params = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate_init': [.1, 0.05, 0.01, 0.005, 0.001]
}

grid_search_cv = GridSearchCV(MLPClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
y_pred = grid_search_cv.predict(X_test)
mln = accuracy_score(y_test, y_pred) * 100

# Results
print("Accuracy for PSV: ", psv)
print("Accuracy for DTC: ", dtc)
print("Accuracy for MLNN: ", mln)
print("El mejor es la Multi-layer Nauronal Network!")

