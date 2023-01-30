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

# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

# 全データ数　28
alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
allnframes = [      -1,         -1,         1,       1,       -1,        -1]
            #  [       4,          6,         5,       5,        4,         4]
            #  [     0~3,        4~9,     10~14,   11~19,    20~23,     24~27]
nframe = -1
# alldatname = ['kobe32']
# allnframes = [      -1]
# alldatname = ['kobe32','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1]
# alldatname = ['traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [        -1,         -1,         1,       1,       -1]


# load data
alldata = []
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
    for meas_num in range(meas.shape[2]):
        data = []
        data.append(meas[:,:,meas_num])
        data.append(mask)
        start_point = meas_num*mask.shape[2]
        data.append(orig[:, :, start_point : start_point + mask.shape[2]])
        alldata.append(data)

data_num = len(alldata)
print(data_num)