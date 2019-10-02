# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
# Common imports
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X_train, y_train)
poly100_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=1, C=5))
])
poly100_kernel_svm_clf.fit(X_train, y_train)
y_pred_d_3 = poly_kernel_svm_clf.predict(X_test)
y_pred_d_10 = poly100_kernel_svm_clf.predict(X_test)
acc_d_3 = accuracy_score(y_test, y_pred_d_3)
acc_d_10 = accuracy_score(y_test, y_pred_d_10)
print("Accuracy for svm degree 3 = ", acc_d_3)
print("Accuracy for svm degree 10 = ", acc_d_10)


# 12. Enter to https://archive.ics.uci.edu/ml/index.php
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data[:, 4:]
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X_train, y_train)
poly100_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=1, C=5))
])
poly100_kernel_svm_clf.fit(X_train, y_train)
y_pred_d_3 = poly_kernel_svm_clf.predict(X_test)
y_pred_d_10 = poly100_kernel_svm_clf.predict(X_test)
acc_d_3 = accuracy_score(y_test, y_pred_d_3)
acc_d_10 = accuracy_score(y_test, y_pred_d_10)
print("Accuracy for svm degree 3 = ", acc_d_3)
print("Accuracy for svm degree 10 = ", acc_d_10)