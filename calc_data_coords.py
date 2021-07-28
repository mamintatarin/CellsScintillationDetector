from subprocess import run
import numpy as np
list_of_coords=[]
x_range=np.arange(0,7)
y_range=np.arange(0,7)
for x in x_range:
    for y in range(x+1):
        list_of_coords.append([x,y])
for arg in list_of_coords:
    result=run(['build/OpNovice','-m','optPhoton.mac','-x',str(arg[0]),'-y',str(arg[1])])
    print(result.stdout)