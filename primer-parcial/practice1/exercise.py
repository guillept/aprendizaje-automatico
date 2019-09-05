'''
Guillermo Pérez Trueba
A01377162
'''

#11. Apply the same process by omitting step 10 to digits, wine, and breast_cancer databases

#DIGITS

# 4.Configure pretty figures and a folder to save them
import matplotlib as mpl
import os

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# 5.Induce a decision tree using wine database
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data[:, 4:]
y = digits.target

# 6.Draw the decision tree model using the following code:

digits_wine = DecisionTreeClassifier(max_depth=4, criterion='entropy')
digits_wine.fit(X, y)
y_pred = digits_wine.predict(X)

from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
    digits_wine,
    out_file=os.path.join(IMAGES_PATH, "digits_tree.dot"),
    rounded=True,
    filled=True
)

Source.from_file(os.path.join(IMAGES_PATH, "digits_tree.dot"))

print("Accuracy for digits is", accuracy_score(y, y_pred) * 100)

#WINE

# 5.Induce a decision tree using wine database
from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data[:,2:]
y = wine.target

# 6.Draw the decision tree model using the following code:

tree_wine = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree_wine.fit(X, y)
y_pred = tree_wine.predict(X)

export_graphviz(
    tree_wine,
    out_file=os.path.join(IMAGES_PATH, "wine_tree.dot"),
    feature_names=wine.feature_names[2:],
    class_names=wine.target_names,
    rounded=True,
    filled=True
)

Source.from_file(os.path.join(IMAGES_PATH, "wine_tree.dot"))

print("Accuracy for wine is", accuracy_score(y, y_pred) * 100)

# BREAST CANCER

# 5.Induce a decision tree using wine database
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data[:, 4:]
y = cancer.target

# 6.Draw the decision tree model using the following code:

cancer_tree = DecisionTreeClassifier(max_depth=4, criterion='entropy')
cancer_tree.fit(X, y)
y_pred = cancer_tree.predict(X)

export_graphviz(
    cancer_tree,
    out_file=os.path.join(IMAGES_PATH, "cancer_tree.dot"),
    rounded=True,
    filled=True
)

Source.from_file(os.path.join(IMAGES_PATH, "cancer_tree.dot"))

print("Accuracy for cancer breast is", accuracy_score(y, y_pred) * 100)
