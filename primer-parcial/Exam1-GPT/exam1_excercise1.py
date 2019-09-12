# Decision tree for regression
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt


np.random.seed(24)
m = 200
X = np.random.rand(m, 1)
y = np.sin(X)
y = y + np.random.randn(m, 1)/100

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X, y)

# Grafica
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees_regression"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
export_graphviz(tree_reg,out_file=os.path.join(IMAGES_PATH, "exercise1.dot"),feature_names=["x1"],rounded=True,filled=True)
