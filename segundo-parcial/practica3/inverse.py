import numpy as np

x = np.array([[4.77, 4.69, 7.14], [4.69, 4.65, 6.92],  [7.14, 6.92, 11.21]])
y = np.linalg.inv(x)
print(y)
