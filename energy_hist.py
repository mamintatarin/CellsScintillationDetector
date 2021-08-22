import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
toDouble=np.float64
dtype=np.dtype([("event","i8"),("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
datalist=[]
PATH='OpNoviceData/unified Te/'
OUT_PATH=PATH+'out/'

def gauss(x,a,x0,sigma):
  return a*np.exp( -(x-x0)**2 / (2*(sigma**2)) )

#построим гистограммы суммы энерговыделений в каждом файле
#Внутри одного файла данные с одной координатой точки обстрела
#Если все поверхности зеркальные, гистограммы суммы для разных файлов (точек) отличаться не должны
for i in range(28):
  tempdata=np.fromfile(PATH+"GammaCamera"+str(i)+".bin",dtype=dtype)
  plt.hist(tempdata['energy1']+tempdata['energy2']+tempdata['energy3']+tempdata['energy4'],bins=50)
  plt.title('x='+str(tempdata['x'][0])+" y="+str(tempdata['y'][0]))
  plt.xlabel('energy, keV')
  plt.ylabel('counts')
  plt.savefig(OUT_PATH+'hist'+str(i))
  plt.clf()
  for j in range(4): #построим распределения энерговыделений в кажом из детекторов для каждой точки обстрела
    hist,edges=np.histogram(tempdata['energy'+str(j+1)],bins=80)
    hist=hist/(hist.sum()*np.diff(edges)[0])
    a_predict=10
    x0_predict=edges[np.argmax(hist)]
    sigma_predict=0.03
    left=int(np.argmax(hist)*0.6)
    right=int(np.argmax(hist)*1.4)
    if right>hist.size-1:
      right=hist.size-1

    popt,pcov = curve_fit(gauss,edges[left:right],hist[left:right],p0=[a_predict,x0_predict,sigma_predict],bounds=(0,[10*a_predict,2*x0_predict,x0_predict]))

    plt.plot(edges[:-1],hist,color='b',label='simulation data')
    plt.plot(edges[left:right],gauss(edges[left:right],popt[0],popt[1],popt[2]),color='red',label='fit')

    plt.title("x0="+np.format_float_positional(popt[1],precision=2)+", sigma = "+np.format_float_positional(popt[2],precision=2))
    plt.xlabel('energy, keV')
    plt.ylabel('counts')
    plt.savefig(OUT_PATH+'hist_'+str(int(tempdata['x'][0]))+str(int(tempdata['y'][0]))+'_'+str(j))
    plt.clf()
  datalist.append(tempdata)

data=np.concatenate(datalist) #склеили все данные из файлов в один массив




hist,edges=np.histogram(data['energy1']+data['energy2']+data['energy3']+data['energy4'],bins=50)
hist=hist/(hist.sum()*np.diff(edges)[0])
#распределение суммарного энерговыделения по всем файлам (точкам обстрела) сразу

  

a_predict=10
x0_predict=edges[np.argmax(hist)]
sigma_predict=0.03

left=int(np.argmax(hist)*0.6)
right=int(np.argmax(hist)*1.4)
if right>hist.size-1:
      right=hist.size-1

popt,pcov = curve_fit(gauss,edges[left:right],hist[left:right],p0=[a_predict,x0_predict,sigma_predict],bounds=(0,[10*a_predict,2*x0_predict,x0_predict]))
plt.plot(edges[:-1],hist,color='b',label='simulation data')
plt.plot(edges[left:right],gauss(edges[left:right],popt[0],popt[1],popt[2]),color='red',label='fit')

plt.title("x0="+np.format_float_positional(popt[1],precision=2)+", sigma/x0 = "+np.format_float_positional(popt[2]/popt[1],precision=2)+", FWHM = "+np.format_float_positional(2.355*popt[2]/popt[1],precision=2))
plt.ylabel('probability density')
plt.legend(loc='best')
plt.xlabel('energy, keV')
plt.savefig(OUT_PATH+'EnergyFit')
plt.clf()
#отрисовка гистограмм окончена


print('Number of events =',data.shape[0])
left_peak=popt[1]-popt[2]
right_peak=popt[1]+popt[2]
data=data[(data['energy1']+data['energy2']+data['energy3']+data['energy4']< right_peak)*(data['energy1']+data['energy2']+data['energy3']+data['energy4']> left_peak)]
print('Number of events inside the peak =',data.shape[0])
#Отсеяли события с аномальными энерговыделениями


result=np.full((7,7,4),np.nan) #квадратики 1 мм, будем хранить тут усредненные энерговыделения для каждой точки обстрела

counts=np.zeros((7,7)) # число событий для каждой точки обстрела

for line in data:
  if(counts[int(line['x']),int(line['y'])]==0):
      result[int(line['x']),int(line['y']),:]=np.asarray([line['energy1'],line['energy2'],line['energy3'],line['energy4']])
      counts[int(line['x']),int(line['y'])]=1
  
  else:   
      result[int(line['x']),int(line['y']),:]=result[int(line['x']),int(line['y']),:]+np.asarray([line['energy1'],line['energy2'],line['energy3'],line['energy4']])
      counts[int(line['x']),int(line['y'])]=counts[int(line['x']),int(line['y'])]+1




#после суммирования надо поделить на количество событий, чтобы получить усредненный сигнал
for i in range(7):
  for j in range(i+1):
    result[i,j,:]=result[i,j,:]/counts[i,j] 
    
    
#Далее с помощью череды симметрий обобщим нашу матрицу в области, которые мы не обстреливали в симуляции
#Нетрудно догадаться, что уникальными являются сигналы в треугольном секторе, занимающем 1/8 всей площади детектора    
matrix=np.full((13,13,4),np.nan)
matrix[6:13,6:13,:]=result #треугольный сектор нам уже известен



for i in range(6,13):
  for j in range(6,i+1):     
     matrix[j,i,:]=np.asarray([matrix[i,j,0],matrix[i,j,3],matrix[i,j,2],matrix[i,j,1]])
for i in range(4):       
   plt.matshow(matrix[:,:,i].T,origin='lower')
   plt.savefig(OUT_PATH+'vis'+str(i))
   plt.clf()   #Если все делаем правильно, картинка должна быть гладкой.
        
for i in range(6,13):
  for j in range(6,13):
     matrix[12-i,12-j,:]=np.asarray([matrix[i,j,2],matrix[i,j,3],matrix[i,j,0],matrix[i,j,1]])
     
for i in range(7,13):
  for j in range(7,13): 
     matrix[12-i,j,:]=np.asarray([matrix[i,j,1],matrix[i,j,0],matrix[i,j,3],matrix[i,j,2]])
     matrix[i,12-j,:]=np.asarray([matrix[i,j,3],matrix[i,j,2],matrix[i,j,1],matrix[i,j,0]])

  
np.save(PATH+'matrix',matrix)

matrix=np.load(PATH+'matrix.npy')    


for i in range(4):#Если все правильно, картинки будут гладкие
  #Честно говоря, я пытался продумать симметричные отражения, сделанные выше (преобразование result в matrix)
  #Но эти картинки все равно получались негладкими, то есть я неправильно делал отражения
  #Это было прямо видно, что кусочек картинки как будто вырезан и повернут
  #В итоге я немного использовал метод тыка, чтобы получить правильную картинку 
  plt.matshow(matrix[:,:,i].T,origin='lower')
  plt.savefig(OUT_PATH+'visMat'+str(i))
  plt.clf()



def predict(e1,e2,e3,e4,matrix,*args):
  #Принимает энерговыделения в детекторе и матрицу усредненных энерговыделений
  #Затем ищет наиболее похожий сигнал в матрице, и возвращает координату этого сигнала
  #args просто для проверок
  centered=np.copy(matrix)
  centered[:,:,0]=centered[:,:,0]-e1
  centered[:,:,1]=centered[:,:,1]-e2
  centered[:,:,2]=centered[:,:,2]-e3
  centered[:,:,3]=centered[:,:,3]-e4
  #Посчитали разность между каждым сигналом в матрице и входным сигналом
  
  mean_photons=matrix/(2.38*(10**(-3)))
  weight = mean_photons**(-1)
  
  minimized_expression = ( weight * (centered**2) ).sum(axis=2)
  ans=np.unravel_index( np.argmin( minimized_expression ) , [matrix.shape[0],matrix.shape[0]] )
  
  if len(args)==2 and args[0]==11 and args[1]==11 and int(ans[0])==int(ans[1])  and int(ans[0])==12:
    print('matrix:',matrix[11,11,:],matrix[12,12,:])
    print('signal:',e1,e2,e3,e4)
    print('Weighted difference 11:',(weight * (centered**2))[11,11])
    print('Weighted difference 12:',(weight * (centered**2))[12,12])
    print('expression:',minimized_expression[11,11],minimized_expression[12,12],'\n')
  
  return ans

        


#Теперь используем те же данные, на которых посчитали средний сигнал, как тестовые
#Эта процедура легальна в нашем случае
#Строим распределения предполагаемых координат для фиксированной точки обстрела
for i in range(0,7):
  for j in range(i+1):    
    dataSlice=data[(data['x'].astype(int)==i)*(data['y'].astype(int)==j)]
    prediction=[]
    for line in dataSlice:
      e1=line['energy1']
      e2=line['energy2']
      e3=line['energy3']
      e4=line['energy4']
      coords=predict(e1,e2,e3,e4,matrix)
      prediction.append([coords[0]-6,coords[1]-6])
    prediction=np.asarray(prediction)  
    
    plt.hist2d(prediction[:,0],prediction[:,1],bins=[np.arange(-6.5,7.5), np.arange(-6.5,7.5)])
    plt.scatter(i,j,color='red')
    plt.title('Probabiillity for each coorinate vs true coordinate')
    plt.xlabel('x,mm')
    plt.ylabel('y,mm')
    plt.colorbar()
    plt.savefig(OUT_PATH+'2dhist/2dhist'+str(i)+str(j))
    plt.clf()
    
   

