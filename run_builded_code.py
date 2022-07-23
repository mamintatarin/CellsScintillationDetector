import subprocess
from multiprocessing import Pool

coordinates=[]
for x in range(-6,6+1):
    for y in range(-6,6+1):
        coordinates.append((x,y))

def foo(args):
    return subprocess.run(["build/OpNovice","-m","optPhoton.mac",'-roof','95','-side','95','-x',str(args[0]),'-y',str(args[1])])
with Pool(5) as pool:
    pool.map(foo, coordinates)

#for arg in coordinates:
#    foo(arg)