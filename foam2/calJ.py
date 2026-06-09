import numpy as np

f1 = np.loadtxt("1/F.txt")
f1 = f1.reshape([3,3])
print(np.linalg.det(f1))

f2 = np.loadtxt("2/F.txt")
f2 = f2.reshape([3,3])
print(np.linalg.det(f2))

f3 = np.loadtxt("3/F.txt")
f3 = f3.reshape([3,3])
print(np.linalg.det(f3))