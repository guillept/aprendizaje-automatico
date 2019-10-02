# Example class

from sklearn.svm import SVC
import numpy as np

X = np.array([[2, 1], [2, -1], [4, 0]]) #suporting vectors
y = np.array([-1, -1, 1]) #alfas

# SVM Classifier model
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)

#SVM support (initial) vectors
print(svm_clf.support_vectors_)
#SVM w vector
print(svm_clf.coef_[0])
#SVM b noise maker
print(svm_clf.intercept_[0])
