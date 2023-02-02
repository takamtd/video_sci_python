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

def read_data(path):
    dirpath = path + "/" + dirname
    filenames = os.listdir(path)
    filenames.sort()
    file_num = len(filenames)
    data = np.zeros((256,256,file_num))
    for i, filename in enumerate(filenames):
        imgs = []
        data[:,:,i] = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)
    return data

def add_noise(y_clear,sigma2):
    noise = np.random.normal(loc = 0, scale = np.sqrt(sigma2), size = y_clear.shape) # sigmaでノイズ生成
    y = y_clear + noise
    return y

def mask_img(origs, masks, noise = False, sigma2=0):
    m = (origs * masks)
    m1 = np.sum(m, axis=2)
    if noise:
        for i in range(m.shape[2]):
            m[:,:,i] = add_noise(m[:,:,i], sigma2)
    m = np.sum(m, axis=2)
    return m

# [0] environment configuration
maskdir = './dataset/cacti/grayscale_benchmark' # dataset
datasetdir = '/home/jovyan/workdir/dataset/cacti/my_dataset/dataset'
resultsdir = './results' # results

# 全データ数　28
# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1,        -1]
            #  [       4,          6,         5,       5,        4,         4]
            #  [     0~3,        4~9,     10~14,   11~19,    20~23,     24~27]
nframe = -1
alldatname = ['kobe32']
allnframes = [      -1]
# alldatname = ['kobe32','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1]
# alldatname = ['traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [        -1,         -1,         1,       1,       -1]

save_path = "/home/jovyan/workdir/dataset/cacti/my_dataset"

# load data
np.random.seed(seed=0)
amount_of_sigma = 5
sigma2 = np.power(255*0.01*amount_of_sigma,2)
alldata = []

matfile = maskdir + '/' + alldatname[0] + '_cacti.mat'
dirnames = os.listdir(datasetdir)
dirnames.sort()
print(dirnames[-6:-5])
# print(dirnames[-4:-2])
print(len(dirnames))
# for dirname in dirnames[-9:]:
for dirname in dirnames[-6:-5]:
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    mask = np.float32(file['mask'])
    orig = np.float32(read_data(datasetdir + "/" + dirname + "/" + "orig/"))
    meas = np.zeros((orig.shape[0], orig.shape[1], int(orig.shape[2]/8)))
    # for i in range(int(orig.shape[2]/8)):
    #     start_point = i * mask.shape[2]
    #     if amount_of_sigma == 0:
    #         meas[:,:,i] = np.float32(mask_img(orig[:, :, start_point : start_point + mask.shape[2]], mask, noise=False))
    #     else:
    #         meas[:,:,i] = np.float32(mask_img(orig[:, :, start_point : start_point + mask.shape[2]], mask, noise=True, sigma2=sigma2))
    for i in range(orig.shape[2]):
        plt.imsave("{}/noise{}/tractor-sand/orig{:03}.bmp".format(save_path, amount_of_sigma, i), add_noise(orig[:,:,i], sigma2), cmap='Greys_r')
    # meas = np.float32(meas)
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

data_num = len(alldata)
print(data_num)