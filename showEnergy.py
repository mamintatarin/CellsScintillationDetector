import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
toDouble=np.float64
dtype=np.dtype([("event","i8"),("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
data=np.fromfile('build/data/'+"GammaCamera.bin",dtype=dtype)
plt.hist(data['energy1']+data['energy2']+data['energy3']+data['energy4'],bins=50)
plt.title('x='+str(data['x'][0])+" y="+str(data['y'][0]))
plt.xlabel('energy, keV')
plt.ylabel('counts')
plt.show()
plt.clf()

