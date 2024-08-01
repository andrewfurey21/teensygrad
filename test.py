import numpy as np

a = np.array([i for i in range(16)])

b = np.reshape(a, (2, 2, 2, 2))

c = np.sum(b, axis=(1, 0, 3, 2))

print(a)
print(b)
print(c)
