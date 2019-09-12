from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_wine = DecisionTreeRegressor(max_depth=5, random_state=42 )
tree_wine.fit(X_train, y_train)
y_pred = tree_wine.predict(X_test)


result=accuracy_score(y_test, y_pred)
print(f'{result*100} %')


