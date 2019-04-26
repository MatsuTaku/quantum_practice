from blueqat import *
import numpy as np

angle = np.linspace(0.0, 2*np.pi, 20)
data = [0 for _ in range(20)]

for i in range(20):
    c = Circuit().ry(angle[i])[0].z[0]
    result = c.run()
    data[i] = np.abs(result[0])*np.abs(result[0]) - np.abs(result[1])*np.abs(result[1])

%matplotlib inline
import matplotlib.pyplot as plt
plt.xlabel('Parameter value')
plt.ylabel('Expectation value')
plt.plot(angle, data)
plt.show()
