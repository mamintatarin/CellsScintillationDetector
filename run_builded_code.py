import subprocess
from multiprocessing import Pool
import numpy as np


use_symmetry = True
step = 0.25 # mm
maxcoord = 6.75 # mm
coordinates = []

if not use_symmetry:
    for x in np.arange(-int(maxcoord/step),int(maxcoord/step)+1)*step:
        for y in np.arange(-int(maxcoord/step),int(maxcoord/step)+1)*step:
            coordinates.append((x,y))
else:
    for x in np.arange(0,int(maxcoord/step)+1)*step:
        for y in np.arange(0,x+step,step):
            coordinates.append((x,y))



def foo(args):
    return subprocess.run(["build/OpNovice","-m","optPhoton.mac",'-roof','95','-side','95','-x',str(args[0]),'-y',str(args[1])])
    print(str(args[0]),str(args[1]))
    
with Pool(5) as pool:
    pool.map(foo, coordinates)

#for arg in coordinates:
#    foo(arg)