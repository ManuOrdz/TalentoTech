import numpy as np
import matplotlib as plt
a = np.array([3,2,])
b = np.array([0,2,1])
if a.shape < b.shape:
    a=np.append(a,1)
print(a*b)
