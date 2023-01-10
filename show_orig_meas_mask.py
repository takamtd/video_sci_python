import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci import admmdenoise_cacti

from scipy.io.matlab.miobase import get_matfile_version
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from utils import (A_, At_)

def logit(p):
        return np.log(p/(1-p))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def show_prams(p):
        return 255*sigmoid(p)

# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1,        -1]
# alldatname = ['drop40','crash32','aerial32']
# allnframes = [       1,       -1,        -1]
alldatname = ['kobe32']
allnframes = [      -1]
count = 0
for datname, nframe in zip(alldatname, allnframes):
    # datname = 'kobe32'        # name of the dataset
    # datname = 'traffic48'     # name of the dataset
    # datname = 'runner40'      # name of the dataset
    # datname = 'drop40'        # name of the dataset
    # datname = 'crash32'       # name of the dataset
    # datname = 'aerial32'      # name of the dataset
    # datname = 'bicycle24'     # name of the dataset
    # datname = 'starfish48'    # name of the dataset

    # datname = 'starfish_c16_48'    # name of the dataset

    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

    # [1] load data
    # if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    meas = np.float32(file['meas'])
    mask = np.float32(file['mask'])
    orig = np.float32(file['orig'])
    # else: # MATLAB .mat v7.3
    #     file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    #     meas = np.float32(file['meas']).transpose()
    #     mask = np.float32(file['mask']).transpose()
    #     orig = np.float32(file['orig']).transpose()

    # print(meas.shape, mask.shape, orig.shape)

    
    SAVE_ORIG = False
    SAVE_MASK = True
    SAVE_MEAS = False

    savedmatdir = resultsdir + '/savedmat/grayscale/' + datname + '/'
    # if not os.path.exists(savedmatdir):
    #     os.makedirs(savedmatdir)
    
    if SAVE_ORIG:
        if not os.path.exists(savedmatdir + 'orig/'):
            os.makedirs(savedmatdir + 'orig/')
        for i in range(orig.shape[2]):
            plt.imsave('{}orig/orig{:02}.jpeg'.format(savedmatdir, i), orig[:,:,i], cmap='Greys_r')
    
    if SAVE_MASK:
        if not os.path.exists(resultsdir + '/savedmat/grayscale/' + 'mask/'):
            os.makedirs(resultsdir + '/savedmat/grayscale/' + 'mask/')
        for i in range(mask.shape[2]):
            plt.imsave(resultsdir + '/savedmat/grayscale/mask/mask{:02}.bmp'.format(i), mask[:,:,i]*255, cmap='Greys_r')

    if SAVE_MEAS:
        if not os.path.exists(savedmatdir + 'meas/'):
            os.makedirs(savedmatdir + 'meas/')
        for i in range(meas.shape[2]):
            plt.imsave('{}meas/meas{:02}.jpeg'.format(savedmatdir, i), meas[:,:,i], cmap='Greys_r')
    
    # 一致判定
    for i in range(mask.shape[2] - 1):
        # flag =  np.allclose(mask, prev_mask)
        sabun = mask[:,:,i+1] - mask[:,:,i]
        # 結果表示
        # print(flag)
        # print(sabun)
    
    count += 1
