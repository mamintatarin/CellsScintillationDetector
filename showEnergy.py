import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import sys
sys.path.insert(0, 'C:/Users/Timur/Desktop/NPM/gamma_detector_tools') # path to gamma_detector_tools project
from gamma_detector_tools import fit_gauss,gaussian
from os import listdir
from os.path import isfile, join
import pandas as pd
from event_location_tools import *

toDouble=np.float64
dtype=np.dtype([("event","i8"),("x",toDouble),("y",toDouble),("energy1",toDouble),("energy2",toDouble),("energy3",toDouble),("energy4",toDouble)])
onlyfiles = [f for f in listdir("data") if isfile(join("data", f))] # get all file names in "data" folder
data_list = []
for f in onlyfiles:
    if "GammaCamera" in f and ".bin" in f:
        data_list.append(np.fromfile(join("data", f),dtype=dtype))
data=np.concatenate(data_list)[['x','y','energy1','energy2','energy3','energy4']] # concatenate data from all files in one array, drop event column


# calculate energy histogram (all photomultipliers summarized)
hist,bins=np.histogram(data['energy1']+data['energy2']+data['energy3']+data['energy4'],bins=50)

# fit hist
amplitude_hist,x0_hist, sigma_hist,baseline_hist = fit_gauss(bins[:-1],hist)

# fit calibrated distribution (to photopeak)
photopeak_position_adc = bins[hist.argmax()] # amplutude corresponding to photopeak in adc counts/not-calibrated amplitudes on photomultipliers
energy = 122 # keV
distribution = hist/(hist.sum()*(bins[1]-bins[0]))
amplitude,x0, sigma,baseline = fit_gauss(bins[:-1]*energy/photopeak_position_adc,distribution)
calibrated_bins = bins[:-1]*energy/photopeak_position_adc # from adc counts to real energy of gamma particle

# show
plt.plot(calibrated_bins,distribution,label='Amplitude distribution')
plt.plot(calibrated_bins,gaussian(calibrated_bins, amplitude, x0, sigma, baseline),label='Gaussian fit')
plt.xlabel('Energy, keV')
plt.title('Gamma energy distribution')
plt.legend()
plt.show()
plt.clf()
plt.plot(bins[:-1],hist,label='Sum distribution')
plt.plot(bins,gaussian(bins, amplitude_hist, x0_hist, sigma_hist, baseline_hist),label='Gaussian fit')
plt.xlabel('Amplitude sum')
plt.title('Photomultipiers sum histogram')
plt.legend()
plt.show()
plt.clf()



# drop everything with energy that is far away from photopeak
deviation = sigma_hist/2
position = bins[hist.argmax()]
eventsnum =  data.shape[0]
data=data[(data['energy1']+data['energy2']+data['energy3']+data['energy4'] < position+deviation )\
    &(data['energy1']+data['energy2']+data['energy3']+data['energy4'] > position-deviation )]
print('events in photopeak:',round(100*data.shape[0]/eventsnum,1),"%")


# to dataframe
dataframe = pd.DataFrame(data=data)
# I ASSUME x AND y BEING INTEGER VALUES IN MILLIMETERS DESPITE THEIR DATATYPE IS FLOAT
# BE CAREFUL WITH THE INPUT DATA 
dataframe['coordinates'] = dataframe.apply(lambda row: [int(row['x']),int(row['y'])], axis=1) # concatenate rows of x and y
true_coordinates = np.asarray(dataframe['coordinates'].to_list())

# fit and predict
#mean_vals, unique_coordinates = fit_mean(dataframe[['energy1','energy2','energy3','energy4']].to_numpy(),true_coordinates)
#predicted_coords = predict_mean(dataframe[['energy1','energy2','energy3','energy4']].to_numpy(), mean_vals, unique_coordinates)
pmt_distributions, pmt_bins, unique_coordinates = fit_max_likelihood(dataframe[['energy1','energy2','energy3','energy4']].to_numpy(),true_coordinates,binsnum=100)
predicted_coords,likelohoods = predict_max_likelihood(dataframe[['energy1','energy2','energy3','energy4']].to_numpy(), pmt_distributions, pmt_bins, unique_coordinates, binsize = 2.38*0.001)

error_matrix = np.zeros((int(np.sqrt(unique_coordinates.shape[0])),int(np.sqrt(unique_coordinates.shape[0]))))
start_x = unique_coordinates[:,0].min()
stop_x = unique_coordinates[:,0].max() 
step_x =(unique_coordinates[:,0].max() - unique_coordinates[:,0].min()) / (np.sqrt(unique_coordinates.shape[0]) -1)

start_y = unique_coordinates[:,1].min()
stop_y = unique_coordinates[:,1].max() 
step_y = (unique_coordinates[:,1].max() - unique_coordinates[:,1].min()) / (np.sqrt(unique_coordinates.shape[0]) -1) 


draw_2d_errors=False
for i,pair in enumerate(unique_coordinates): # draw error distributions for every unique point

    indices = np.nonzero(true_coordinates == (pair[0],pair[1])) # select events with particular coordinate pair
    mean_error_x = abs(predicted_coords[indices,0] - true_coordinates[indices,0])
    mean_error_y = abs(predicted_coords[indices,1] - true_coordinates[indices,1])
    bins_x,hist_x=np.unique(mean_error_x,return_counts=True)
    bins_y,hist_y=np.unique(mean_error_y,return_counts=True)
    fwhm_x = find_fwhm_precision(bins_x, hist_x)[1]
    fwhm_y = find_fwhm_precision(bins_y, hist_y)[1]
    fwhm = np.sqrt(fwhm_x**2 + fwhm_y**2)

    error_matrix[int((pair[0]-start_x)/step_x),int((pair[1]-start_y)/step_y)] = fwhm

    if draw_2d_errors:
        plt.plot(bins_x,hist_x,label='x error')
        plt.plot(bins_y,hist_y,label='y error')
        plt.xlabel('Error, mm')
        plt.ylabel('Density')
        plt.legend()
        plt.title('x='+str(pair[0])+' mm, y='+str(pair[1])+' mm, FWHM='+str(round(fwhm,2)))
        plt.show()
        plt.clf()    


plt.matshow(error_matrix)
plt.colorbar()
plt.show()
plt.clf()






