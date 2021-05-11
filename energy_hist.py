import matplotlib.pyplot as plt
import numpy as np
toDouble=np.float64
dtype=np.dtype([("event","i8"),("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
datalist=[]






for i in range(28):
  tempdata=np.fromfile("GammaCamera"+str(i)+".bin",dtype=dtype)
  plt.hist(tempdata['energy1']+tempdata['energy2']+tempdata['energy3']+tempdata['energy4'],bins=50)
  plt.title('x='+str(tempdata['x'][0])+" y="+str(tempdata['x'][1]))
  plt.xlabel('energy, MeV')
  plt.ylabel('counts')
  plt.savefig('hist'+str(i))
  plt.clf()
  datalist.append(tempdata)

data=np.concatenate(datalist)

result=np.full((7,7,4),np.nan)
counts=np.zeros((7,7))

for line in data:
   if(counts[int(line['x']),int(line['y'])]==0):
      result[int(line['x']),int(line['y']),:]=np.asarray([line['energy1'],line['energy2'],line['energy3'],line['energy4']])
      counts[int(line['x']),int(line['y'])]=1
   else:   
      result[int(line['x']),int(line['y']),:]=result[int(line['x']),int(line['y']),:]+np.asarray([line['energy1'],line['energy2'],line['energy3'],line['energy4']])
      counts[int(line['x']),int(line['y'])]=counts[int(line['x']),int(line['y'])]+1




for i in range(7):
  for j in range(i+1):
    result[i,j,:]=result[i,j,:]/counts[i,j]
    
    

    

matrix=np.full((13,13,4),np.nan)
matrix[6:13,6:13,:]=result



for i in range(6,13):
  for j in range(6,i+1):     
     matrix[j,i,:]=np.asarray([matrix[i,j,0],matrix[i,j,3],matrix[i,j,2],matrix[i,j,1]])
for i in range(4):       
   plt.matshow(matrix[:,:,i].T,origin='lower')
   plt.savefig('vis'+str(i))
   plt.clf()   
     

   


for i in range(6,13):#not ready
  for j in range(6,13):
     matrix[12-i,12-j,:]=np.asarray([matrix[i,j,2],matrix[i,j,3],matrix[i,j,0],matrix[i,j,1]])
     
for i in range(7,13):
  for j in range(7,13): 
     matrix[12-i,j,:]=np.asarray([matrix[i,j,1],matrix[i,j,0],matrix[i,j,3],matrix[i,j,2]])
     matrix[i,12-j,:]=np.asarray([matrix[i,j,3],matrix[i,j,2],matrix[i,j,1],matrix[i,j,0]])

  
    

for i in range(4):       
   plt.matshow(matrix[:,:,i].T,origin='lower')
   plt.savefig('visMat'+str(i))
   plt.clf()
    
  


