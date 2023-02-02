import os
import time
import math
import h5py
import csv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
import cv2

def add_noise(y_clear,sigma2):
    noise = np.random.normal(loc = 0, scale = np.sqrt(sigma2), size = y_clear.shape) 
    y = y_clear + noise
    return y

# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

# 全データ数　28
# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1,        -1]
            #  [       4,          6,         5,       5,        4,         4]
            #  [     0~3,        4~9,     10~14,   11~19,    20~23,     24~27]
nframe = -1
alldatname = ['traffic48']
allnframes = [      -1]
# alldatname = ['kobe32','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1]
# alldatname = ['traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [        -1,         -1,         1,       1,       -1]

save_path = "/home/jovyan/workdir/dataset/cacti/my_dataset"
# load data
alldata = []
np.random.seed(seed=0)
amount_of_sigma = 5
sigma2 = np.power(255*0.01*amount_of_sigma,2)
for datname, nframe in zip(alldatname, allnframes):
    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

    # if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    meas = np.float32(file['meas'])
    mask = np.float32(file['mask'])
    orig = np.float32(file['orig'])
    # meas = torch.from_numpy(meas).to(device)
    # mask = torch.from_numpy(mask).to(device)
    # orig = torch.from_numpy(orig).to(device)
    # for meas_num in range(meas.shape[2]):
    #     data = []
    #     data.append(meas[:,:,meas_num])
    #     data.append(mask)
    #     start_point = meas_num*mask.shape[2]
    #     data.append(orig[:, :, start_point : start_point + mask.shape[2]])
    #     alldata.append(data)

    for i in range(orig.shape[2]):
        plt.imsave("{}/noise{}/traffic48/orig{:03}.bmp".format(save_path, amount_of_sigma, i), add_noise(orig[:,:,i], sigma2), cmap='Greys_r')

data_num = len(alldata)
print(data_num)