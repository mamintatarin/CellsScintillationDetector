import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import sys
sys.path.insert(0, 'C:/Users/Timur/Desktop/NPM/gamma_detector_tools') # path to gamma_detector_tools project
from gamma_detector_tools import fit_gauss,gaussian,reflect_data
from os import listdir
from os.path import isfile, join
from event_location_tools import *
from tqdm import tqdm



toDouble=np.float64
data_path = "data/launch4_2mm"
dtype=np.dtype([("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))] # get all file names in "data" folder
data_list = []
for f in onlyfiles:
    if "GammaCamera" in f and ".bin" in f:
        data_list.append(np.fromfile(join(data_path, f),dtype=dtype))
data=np.concatenate(data_list) # concatenate data from all files in one array
# calculate energy histogram (all photomultipliers summarized)
hist,bins=np.histogram(data['energy1']+data['energy2']+data['energy3']+data['energy4'],bins=50)

# fit hist
amplitude_hist, x0_hist, sigma_hist, baseline_hist = fit_gauss(bins[:-1],hist,max_region=(bins[int(bins.shape[0]/2)],bins.max()))

# fit calibrated distribution (to photopeak)
energy = 122 # keV
photopeak_position_adc = x0_hist
distribution = hist/(hist.sum()*(bins[1]-bins[0]))
amplitude,x0, sigma,baseline = fit_gauss(bins[:-1]*energy/photopeak_position_adc,distribution,max_region=(100,140))
calibrated_bins = bins[:-1]*energy/photopeak_position_adc # from adc counts to real energy of gamma particle

# drop everything with energy that is far away from photopeak
deviation = sigma_hist*0.5
position = photopeak_position_adc
eventsnum =  data.shape[0]
data=data[(data['energy1']+data['energy2']+data['energy3']+data['energy4'] < position+deviation )\
    &(data['energy1']+data['energy2']+data['energy3']+data['energy4'] > position-deviation )]
print('events in photopeak:',round(100*data.shape[0]/eventsnum,1),"%")

# show
plt.plot(calibrated_bins,distribution,label='Amplitude distribution')
plt.plot(calibrated_bins,gaussian(calibrated_bins, amplitude, x0, sigma, baseline),label='Gaussian fit')
plt.vlines([(position-deviation)*energy/photopeak_position_adc,(position+deviation)*energy/photopeak_position_adc],
           distribution.min(),distribution.max(),color='red',label='Photopeak')
plt.xlabel('Energy, keV')
plt.title('$\gamma$-energy, calculated by total signal')
plt.legend()
plt.show()
plt.clf()
plt.plot(bins[:-1],hist,label='Sum distribution')
plt.plot(bins,gaussian(bins, amplitude_hist, x0_hist, sigma_hist, baseline_hist),label='Gaussian fit')
plt.vlines([position-deviation,position+deviation],hist.min(),hist.max(),color='red',label='Photopeak')
plt.xlabel('Amplitude sum')
plt.title('Photomultipiers sum histogram')
plt.legend()
plt.show()
plt.clf()



#data= reflect_data(data)
true_coords = np.zeros((data.shape[0],2))
true_coords[:,0]=data['x']
true_coords[:,1]=data['y']
energies = np.zeros((data.shape[0],4))
energies[:,0]= data['energy1']
energies[:,1]= data['energy2']
energies[:,2]= data['energy3']
energies[:,3]= data['energy4']


#this is only for analytical solution

abs_len = 2.33 #mm
z_height = 2
z_grid = np.arange(0,100)*(z_height/100)
median_z = z_height - (z_grid*np.exp(-z_grid/abs_len)).sum() / (np.exp(-z_grid/abs_len).sum())
print('median z = ',median_z)
center_data = data[(data['x']==0)&(data['y']==0)]
mean_signal = sum([center_data['energy'+str(i)] for i in range(1,5)]).mean()
intencity = 122*(40*2.38/1000)
points_of_scintillation = np.zeros((1,3))
points_of_scintillation[:,2] = median_z 
analytical_signal_center,pmt_coords = analytical_intencity(points_of_scintillation, (0.95,0.95,0.95,0.95,0.95), (14,14,z_height), (2,2), (50,50), intencity)
analytical_signal_center = analytical_signal_center.sum(axis=1).mean()*0.3
calibration = mean_signal/analytical_signal_center

unique_coordinates = np.unique(true_coords, axis=0)
mean_analytical_results, pmt_coords = fit_mean_analytical(unique_coordinates,
                                (0.95,0.95,0.95,0.95,0.95),(14,14,z_height), (2,2), (50,50),intencity*calibration,median_z,0.3)
for i in tqdm(range(unique_coordinates.shape[0])):
    temp_sig = mean_analytical_results[i,:].copy() 
    mean_analytical_results[i,0] = temp_sig[np.nonzero((pmt_coords[:,0]==3.5)&(pmt_coords[:,1]==3.5))]
    mean_analytical_results[i,1] = temp_sig[np.nonzero((pmt_coords[:,0]==-3.5)&(pmt_coords[:,1]==3.5))]
    mean_analytical_results[i,2] = temp_sig[np.nonzero((pmt_coords[:,0]==-3.5)&(pmt_coords[:,1]==-3.5))]
    mean_analytical_results[i,3] = temp_sig[np.nonzero((pmt_coords[:,0]==3.5)&(pmt_coords[:,1]==-3.5))]

# reflect 1/8 of rectangle to the whole region
temp = np.zeros(unique_coordinates.shape[0],dtype=dtype)
temp['x']=unique_coordinates[:,0]
temp['y']=unique_coordinates[:,1]
temp['energy1']=mean_analytical_results[:,0]
temp['energy2']=mean_analytical_results[:,1]
temp['energy3']=mean_analytical_results[:,2]
temp['energy4']=mean_analytical_results[:,3]
temp = reflect_data(temp)
mean_analytical_results = np.zeros((temp.shape[0],4))
mean_analytical_results[:,0]=temp['energy1']
mean_analytical_results[:,1]=temp['energy2']
mean_analytical_results[:,2]=temp['energy3']
mean_analytical_results[:,3]=temp['energy4']
unique_coordinates = np.zeros((temp.shape[0],2))
unique_coordinates[:,0]=temp['x']
unique_coordinates[:,1]=temp['y']

predicted_coords = predict_mean(energies, mean_analytical_results, unique_coordinates)

#Anger method
#predicted_coords = predict_anger(energies, np.asarray([[3.5,3.5],[-3.5,3.5],[-3.5,-3.5],[3.5,-3.5]])) 
#unique_coordinates = np.unique(true_coords, axis=0)
# fit and predict by averaging
"""
mean_vals, unique_coordinates = fit_mean(energies,true_coords)
# reflect 1/8 of rectangle to the whole region
temp = np.zeros(unique_coordinates.shape[0],dtype=dtype)
temp['x']=unique_coordinates[:,0]
temp['y']=unique_coordinates[:,1]
temp['energy1']=mean_vals[:,0]
temp['energy2']=mean_vals[:,1]
temp['energy3']=mean_vals[:,2]
temp['energy4']=mean_vals[:,3]
temp = reflect_data(temp)
mean_vals = np.zeros((temp.shape[0],energies.shape[1]))
mean_vals[:,0]=temp['energy1']
mean_vals[:,1]=temp['energy2']
mean_vals[:,2]=temp['energy3']
mean_vals[:,3]=temp['energy4']
unique_coordinates = np.zeros((temp.shape[0],2))
unique_coordinates[:,0]=temp['x']
unique_coordinates[:,1]=temp['y']
predicted_coords = predict_mean(energies, mean_vals, unique_coordinates)

"""
#x_shape = (np.unique(unique_coordinates[:,0]).shape[0]-1)*2 + 1
#y_shape = (np.unique(unique_coordinates[:,1]).shape[0]-1)*2 + 1
x_shape = np.unique(unique_coordinates[:,0]).shape[0]
y_shape = np.unique(unique_coordinates[:,1]).shape[0]
error_matrix = np.zeros((x_shape,y_shape))
systematic_error_matrix = np.zeros((x_shape,y_shape))
start_x = unique_coordinates[:,0].min()
stop_x = unique_coordinates[:,0].max() 
step_x =(unique_coordinates[:,0].max() - unique_coordinates[:,0].min()) / (np.unique(unique_coordinates[:,0]).shape[0] -1)
start_y = unique_coordinates[:,1].min()
stop_y = unique_coordinates[:,1].max() 
step_y = (unique_coordinates[:,1].max() - unique_coordinates[:,1].min()) / (np.unique(unique_coordinates[:,1]).shape[0] -1) 
from scipy.stats import mode
draw_distr = True
for i,pair in enumerate(unique_coordinates): # draw error distributions for every unique point
    if pair[0]>=pair[1] and pair[0]>=0 and pair[1]>=0:
        indices = np.nonzero((true_coords[:,0] ==pair[0]) & (true_coords[:,1]==pair[1])) # select events with particular coordinate pair
  
        error_x = predicted_coords[indices[0],0] - true_coords[indices[0],0]
        error_y = predicted_coords[indices[0],1] - true_coords[indices[0],1]
        error_2d = np.zeros((indices[0].shape[0],2))
        error_2d[:,0] = predicted_coords[indices[0],0] - true_coords[indices[0],0]
        error_2d[:,1] = predicted_coords[indices[0],1] - true_coords[indices[0],1]
        bins_x,hist_x=np.unique(error_x,return_counts=True)
        bins_y,hist_y=np.unique(error_y,return_counts=True)
        
        syst_x = mode(error_x, axis=None)[0][0]
        syst_y = mode(error_y, axis=None)[0][0]
        syst = np.sqrt((syst_x**2+syst_y**2)/2)
        #syst = np.mean(error_2d)
        systematic_error_matrix[int((pair[0]-start_x)/step_x),int((pair[1]-start_y)/step_y)] = syst
        
        fwhm_x = np.std(error_x-mode(error_x, axis=None)[0][0])*2.355
        fwhm_y = np.std(error_x-mode(error_x, axis=None)[0][0])*2.355
        fwhm = np.sqrt((fwhm_x**2+fwhm_y**2)/2)
        #fwhm = np.std(error_2d)*2.355
        error_matrix[int((pair[0]-start_x)/step_x),int((pair[1]-start_y)/step_y)] = fwhm
        
        if draw_distr and pair[0]==pair[1] and pair[0]==0:
            plt.plot(bins_x,hist_x,label='x-axis')
            plt.plot(bins_y,hist_y,label='y-axis')
            plt.title('(x,y) = ('+str(pair[0])+','+str(pair[1])+")")
            plt.xlabel('Error, mm')
            plt.ylabel('Events number')
            plt.legend()
            plt.show()
            plt.clf()
            

for i in range(error_matrix.shape[0]):
    for j in range(error_matrix.shape[1]):
        if error_matrix[i,j]!=0:
            error_matrix[j,i]=error_matrix[i,j]
            error_matrix[error_matrix.shape[0]-i-1,j]=error_matrix[i,j]
            error_matrix[i,error_matrix.shape[1]-j-1]=error_matrix[i,j]
            error_matrix[error_matrix.shape[0]-i-1,error_matrix.shape[1]-j-1]=error_matrix[i,j]
            error_matrix[j,error_matrix.shape[0]-i-1]=error_matrix[i,j]
            error_matrix[error_matrix.shape[1]-j-1,i]=error_matrix[i,j]
            error_matrix[error_matrix.shape[1]-j-1,error_matrix.shape[0]-i-1]=error_matrix[i,j]
        if systematic_error_matrix[i,j]!=0:
            systematic_error_matrix[j,i]=systematic_error_matrix[i,j]
            systematic_error_matrix[systematic_error_matrix.shape[0]-i-1,j]=systematic_error_matrix[i,j]
            systematic_error_matrix[i,systematic_error_matrix.shape[1]-j-1]=systematic_error_matrix[i,j]
            systematic_error_matrix[systematic_error_matrix.shape[0]-i-1,systematic_error_matrix.shape[1]-j-1]=systematic_error_matrix[i,j]
            systematic_error_matrix[j,systematic_error_matrix.shape[0]-i-1]=systematic_error_matrix[i,j]
            systematic_error_matrix[systematic_error_matrix.shape[1]-j-1,i]=systematic_error_matrix[i,j]
            systematic_error_matrix[systematic_error_matrix.shape[1]-j-1,systematic_error_matrix.shape[0]-i-1]=systematic_error_matrix[i,j]

plt.matshow(error_matrix.T)
plt.title('Resolution (FWHM), mm.')
plt.xlabel('x coordinate, mm')
plt.ylabel('y coordinate, mm')
plt.colorbar()
# draw every n-th tick with the shift of m element from zero position
n = 4
m = 1
if n!=1:
    print('Warning! Some xticks and yticks will be skipped due to n!=1')
if m!=0:
    print('Warning! Some xticks and yticks will be skipped due to m!=0')

second_arg_x = np.arange(int((stop_x-start_x)/step_x)+1)*step_x+start_x
second_arg_y = np.arange(int((stop_y-start_y)/step_y)+1)*step_y+start_y

plt.xticks([i*n+m for i in range(int((second_arg_x.shape[0]-1-m)/n)+1)], 
           [second_arg_x[i*n+m] for i in range(int((second_arg_x.shape[0]-1-m)/n)+1)])
plt.yticks([i*n+m for i in range(int((second_arg_y.shape[0]-1-m)/n)+1)], 
           [second_arg_y[i*n+m] for i in range(int((second_arg_y.shape[0]-1-m)/n)+1)])
plt.clim(0, 2.3) 
plt.show()
plt.clf()
plt.matshow(systematic_error_matrix.T)
plt.title('Systematic error, mm.')
plt.xlabel('x coordinate, mm')
plt.ylabel('y coordinate, mm')
plt.colorbar()

plt.xticks([i*n+m for i in range(int((second_arg_x.shape[0]-1-m)/n)+1)], 
           [second_arg_x[i*n+m] for i in range(int((second_arg_x.shape[0]-1-m)/n)+1)])
plt.yticks([i*n+m for i in range(int((second_arg_y.shape[0]-1-m)/n)+1)], 
           [second_arg_y[i*n+m] for i in range(int((second_arg_y.shape[0]-1-m)/n)+1)])
plt.clim(0, 1.6) 
plt.show()
plt.clf()






