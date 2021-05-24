import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
toDouble=np.float64
dtype=np.dtype([("event","i8"),("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
datalist=[]
PATH='Variant2/'
OUT_PATH=PATH+'out/'




for i in range(28):
  tempdata=np.fromfile(PATH+"GammaCamera"+str(i)+".bin",dtype=dtype)
  plt.hist(tempdata['energy1']+tempdata['energy2']+tempdata['energy3']+tempdata['energy4'],bins=50)
  plt.title('x='+str(tempdata['x'][0])+" y="+str(tempdata['x'][1]))
  plt.xlabel('energy, MeV')
  plt.ylabel('counts')
  plt.savefig(OUT_PATH+'hist'+str(i))
  plt.clf()
  datalist.append(tempdata)

data=np.concatenate(datalist)




hist,edges=np.histogram(data['energy1']+data['energy2']+data['energy3']+data['energy4'],bins=50)
hist=hist/(hist.sum()*np.diff(edges)[0])

def gauss(x,a,x0,sigma):
  return a*np.exp(-(x-x0)**2/(2*sigma**2))
  

a_predict=2500
x0_predict=edges[np.argmax(hist)]
#sigma_predict=0.0002
sigma_predict=0.00008

left=int(np.argmax(hist)*0.8)
right=int(np.argmax(hist)*1.4)
popt,pcov = curve_fit(gauss,edges[left:right],hist[left:right],p0=[a_predict,x0_predict,sigma_predict])
plt.plot(edges[:-1],hist,color='b',label='simulation data')
plt.plot(edges[left:right],gauss(edges[left:right],popt[0],popt[1],popt[2]),color='red',label='fit')
plt.title("Sigma/x0 = "+str(popt[2]/popt[1])+", FWHM = "+str(2.355*popt[2]/popt[1]))
plt.ylabel('probabillity')
plt.legend(loc='best')
plt.xlabel('energy, MeV')
plt.savefig(OUT_PATH+'EnergyFit')
plt.clf()

data=data[(data['energy1']+data['energy2']+data['energy3']+data['energy4']< popt[2]+popt[1])*(data['energy1']+data['energy2']+data['energy3']+data['energy4']> popt[1]-popt[2])]

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
   plt.savefig(OUT_PATH+'vis'+str(i))
   plt.clf()   
     

   


for i in range(6,13):
  for j in range(6,13):
     matrix[12-i,12-j,:]=np.asarray([matrix[i,j,2],matrix[i,j,3],matrix[i,j,0],matrix[i,j,1]])
     
for i in range(7,13):
  for j in range(7,13): 
     matrix[12-i,j,:]=np.asarray([matrix[i,j,1],matrix[i,j,0],matrix[i,j,3],matrix[i,j,2]])
     matrix[i,12-j,:]=np.asarray([matrix[i,j,3],matrix[i,j,2],matrix[i,j,1],matrix[i,j,0]])

  
np.save(PATH+'matrix',matrix)

matrix=np.load(PATH+'matrix.npy')    


for i in range(4):       
   plt.matshow(matrix[:,:,i].T,origin='lower')
   plt.savefig(OUT_PATH+'visMat'+str(i))
   plt.clf()


def predict(e1,e2,e3,e4,matrix):
   centered=np.copy(matrix)
   centered[:,:,0]=centered[:,:,0]-e1
   centered[:,:,1]=centered[:,:,1]-e2
   centered[:,:,2]=centered[:,:,2]-e3
   centered[:,:,3]=centered[:,:,3]-e4
   return np.unravel_index(np.argmin(centered[:,:,0]**2+centered[:,:,1]**2+centered[:,:,2]**2+centered[:,:,3]**2),(13,13))
        




print(popt[1]-popt[2],popt[2]+popt[1])
for i in range(0,7):
  for j in range(i+1):    
    dataSlice=data[(data['x'].astype(np.int)==i)*(data['y'].astype(np.int)==j)]
    prediction=[]
    for line in dataSlice:
      e1=line['energy1']
      e2=line['energy2']
      e3=line['energy3']
      e4=line['energy4']
      coords=predict(e1,e2,e3,e4,matrix)
      prediction.append([int(coords[0]),int(coords[1])])
    prediction=np.asarray(prediction)  
    plt.hist2d(prediction[:,0],prediction[:,1])
    plt.title('x='+str(i+6)+' y='+str(j+6))
    plt.xlabel('x,mm')
    plt.ylabel('y,mm')
    plt.colorbar()
    plt.savefig(OUT_PATH+'2dhist/2dhist'+str(i)+str(j))
    plt.clf()
    

