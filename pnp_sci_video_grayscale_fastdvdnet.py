import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
import cv2

from pnp_sci import admmdenoise_cacti

from scipy.io.matlab.miobase import get_matfile_version
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import pandas as pd
import math
import torch
from packages.fastdvdnet.models import FastDVDnet

from utils import (A_, At_)

def logit(p):
    return np.log(p/(1-p))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def show_prams(p):
    return 255*sigmoid(p)

def relu(x):
    return max(0, x)

def add_noise(y_clear,sigma2):
    noise = np.random.normal(loc = 0, scale = y_clear.shape[2]*np.sqrt(sigma2), size = y_clear.shape) # sigmaでノイズ生成
    y = y_clear + noise # ブラー画像にノイズを不可
    return y

# [0] environment configuration

alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
allnframes = [      -1,         -1,        -1,      -1,       -1,        -1]
# alldatname = ['drop40','crash32','aerial32']
# allnframes = [       1,       -1,        -1]
# alldatname = ['runner40', 'drop40']
# allnframes = [        -1,       -1] 
# alldatname = ['traffic48']
# allnframes = [      -1]
count = 0

MAXB = 255.


# In[6]:
#################################################################################

## GAP-FastDVDnet
# projmeth = 'gap' # projection method
projmeth = 'gap'
tv_initialize = False
_lambda = 1 # regularization factor
accelerate = False # enable accelerated version of GAP
train_delta = False
train_gamma = False
denoiser = 'fastdvdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP

method_type = 9
OPTION = True
SAVE_RESULT = False
SAVE_DATA = True
SAVE_MEAS = False

# ノイズの設定
np.random.seed(seed=0)
amount_of_sigma = 5
sigma2 = np.power(255*0.01*amount_of_sigma,2)

if method_type == 1:
    sigma    = [50/255, 25/255, 12/255] # pre-set noise standard deviation
    iter_max = [20, 20, 20] # maximum number of iterations
    if accelerate:
        if amount_of_sigma == 0:
            policy_name = 'method1_acc'
        else:
            policy_name = 'method1_acc_add_meas_noise{}'.format(amount_of_sigma)
    else:
        if amount_of_sigma == 0:
            policy_name = 'method1'
        else:
            policy_name = 'method1_add_meas_noise{}'.format(amount_of_sigma)
elif method_type == 2:
    sigma    = [50*0.97**i/255 for i in range(60)]
    iter_max = [1 for i in range(60)]
elif method_type == 3:
    sigma    = [12/255]
    iter_max = [60]
    if accelerate:
        if amount_of_sigma == 0:
            policy_name = 'fixed12_acc'
        else:
            policy_name = 'fixed12_acc_add_meas_noise'
    else:
        if amount_of_sigma == 0:
            policy_name = 'fixed12'
        else:
            policy_name = 'fixed12_add_meas_noise'
elif method_type == 4:
    sigma    = [50/255]
    iter_max = [60]
    if accelerate:
        if amount_of_sigma == 0:
            policy_name = 'fixed50_acc'
        else:
            policy_name = 'fixed50_acc_add_meas_noise'
    else:
        if amount_of_sigma == 0:
            policy_name = 'fixed50'
        else:
            policy_name = 'fixed50_add_meas_noise'
elif method_type == 5:
    sigma    = [(50*0.5**(i/40))/255 for i in range(80)]
    iter_max = [1 for i in range(80)]
elif method_type == 6:
    sigma    = [(50*0.5**(i/50))/255 for i in range(80)]
    iter_max = [1 for i in range(80)]
elif method_type == 7:
    sigma = [(42/(1 + np.exp((i-30)/10)) + 12)/255 for i in range(80)]
    iter_max = [1 for i in range(80)]
elif method_type == 8:
    sigma    = [(100*0.5**(i/20))/255 for i in range(80)]
    iter_max = [1 for i in range(80)]
elif method_type == 9:
    policy_name = 'davis_train_add_noise'
    parameter_name = 'sigma'
    # policy_name = 'kobe_method1'
    

    dir_path = "/home/jovyan/workdir/results/" + "trainning_data/" + projmeth + '/'
    filename = parameter_name + '_' + policy_name
    file_path = dir_path + "data_files/" + filename + ".csv"
    data = pd.read_csv(file_path, usecols=[1], header=None)
    data = pd.Series(data[1])
    data = data.str.replace('\[','')
    data = data.str.replace('\]','')
    data = data.str.replace('\[Parameter containing:\ntensor\(\[','')
    data = data.str.replace('\[Parameter containing:\ntensor\(\[ ','')
    data = data.str.replace('\], device=\'cuda:0\',\n       requires_grad=True\)\]','')
    data = data.str.replace('\],\n       device=\'cuda:0\', requires_grad=True\)\]','')
    data = data.str.replace('\n       ','')
    data = data.str.split(', ', expand=True)
    data = data.astype(float)
    sigma = [sigmoid(d) for d in data.iloc[-1].tolist()]

    if train_delta == True:
        parameter_name = 'delta'
        dir_path = "/home/jovyan/workdir/results/" + "trainning_data/" + projmeth + '/'
        filename = parameter_name + '_' + policy_name
        file_path = dir_path + "data_files/" + filename + ".csv"
        data = pd.read_csv(file_path, usecols=[1], header=None)
        data = pd.Series(data[1])
        data = data.str.replace('\[','')
        data = data.str.replace('\]','')
        data = data.str.split(', ', expand=True)
        data = data.astype(float)
        delta = [relu(d) for d in data.iloc[-1].tolist()]
    
    if train_gamma == True:
        parameter_name = 'gamma'
        dir_path = "/home/jovyan/workdir/results/" + "trainning_data/" + projmeth + '/'
        filename = parameter_name + '_' + policy_name
        file_path = dir_path + "data_files/" + filename + ".csv"
        data = pd.read_csv(file_path, usecols=[1], header=None)
        data = pd.Series(data[1])
        data = data.str.replace('\[','')
        data = data.str.replace('\]','')
        data = data.str.split(', ', expand=True)
        data = data.astype(float)
        gamma = [relu(d) for d in data.iloc[-1].tolist()]

    # print(sigma)
    iter_max = [1 for i in range(60)]

useGPU = True # use GPU

datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

# pre-load the model for fastdvdnet image denoising
NUM_IN_FR_EXT = 5 # temporal size of patch
model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

# Load saved weights
state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
if useGPU:
    device_ids = [0]
    # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    model = model.cuda()
# else:
    # # CPU mode: remove the DataParallel wrapper
    # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

model.load_state_dict(state_temp_dict)

# Sets the model in evaluation mode (e.g. it removes BN)
model.eval()

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
    print()
    print(datname)
    print()

    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

    # In[3]:
    # from scipy.io.matlab.mio import _open_file

    # [1] load data
    # if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    meas = np.float32(file['meas'])
    mask = np.float32(file['mask'])
    orig = np.float32(file['orig'])
    if amount_of_sigma != 0:
        meas = add_noise(meas, sigma2)
    # else: # MATLAB .mat v7.3
    #     file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    #     meas = np.float32(file['meas']).transpose()
    #     mask = np.float32(file['mask']).transpose()
    #     orig = np.float32(file['orig']).transpose()
    
    
    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    A  = lambda x :  A_(x, mask) # forward model function handle
    At = lambda y : At_(y, mask) # transpose of forward model

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1

    iframe = 0
    if nframe < 0:
        nframe = meas.shape[2]
    print(meas.shape)
    if projmeth.lower() == 'gap':
        if tv_initialize:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=vgaptv, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
        elif train_delta:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,  delta=delta,
                                                denoiser=denoiser, model=model,
                                                iter_max=iter_max, sigma=sigma)
        else:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
    elif projmeth.lower() == 'admm':
        if tv_initialize:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=vgaptv, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
        elif train_gamma:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda,  gamma=gamma,
                                                denoiser=denoiser, model=model,
                                                iter_max=iter_max, sigma=sigma)
        else:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
    else:
        print('Unsupported projection method %s' % projmeth.upper())
    
    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapfastdvdnet), mean(ssim_gapfastdvdnet), tgapfastdvdnet))


    # In[8]:
    # [3.3] result demonstration of GAP-Denoise
    nmask = mask.shape[2]
    
    if OPTION:
        if method_type == 9 and amount_of_sigma != 0:
            option_name = policy_name + '_add_meas_noise{}'.format(amount_of_sigma)
        else:
            option_name = policy_name

    savedmatdir = resultsdir + '/savedmat/grayscale/' + projmeth + '/'+ option_name + '/' + datname + '/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)
    
    # sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
    #             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
    if SAVE_RESULT:
        savedir = savedmatdir + "recon_imgs" + "/" # + 'data/csv_folder/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i in range(orig.shape[2]):
            # plt.imsave(savedir + '{:03}.bmp'.format(i), vgapfastdvdnet[:,:,i]*255, cmap='Greys_r')
            cv2.imwrite(savedir + '{:03}.bmp'.format(i), vgapfastdvdnet[:,:,i]*255)
        
    if SAVE_DATA:
        savedir = savedmatdir # + 'data/csv_folder/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        psnr_file = savedir
        psnrall_file = savedir
        ssim_file = savedir
        if tv_initialize:
            psnr_file += "psnr_method{:d}_tv_initialize{:d}.csv".format(method_type, iter_max)
            psnrall_file += "psnrall_method{:d}_tv_initialize{:d}.csv".format(method_type, iter_max)
            ssim_file += "ssim_method{:d}_tv_initialize{:d}.csv".format(method_type, iter_max)
        elif OPTION:
            psnr_file += "psnr_{}.csv".format(option_name)
            psnrall_file += "psnrall_{}.csv".format(option_name)
            ssim_file += "ssim_{}.csv".format(option_name)
        else:
            psnr_file += "psnr_method{:d}.csv".format(method_type)
            psnrall_file += "psnrall_method{:d}.csv".format(method_type)
            ssim_file += "ssim_method{:d}.csv".format(method_type)
            
        psnr_gapfastdvdnet = np.array(psnr_gapfastdvdnet)
        psnrall_gapfastdvdnet = np.array(psnrall_gapfastdvdnet)
        ssim_gapfastdvdnet = np.array(ssim_gapfastdvdnet)
        psnrmean_gapfastdvdnet = psnrall_gapfastdvdnet.mean(axis=0)
        np.savetxt(psnr_file, psnr_gapfastdvdnet)
        np.savetxt(psnrall_file, psnrmean_gapfastdvdnet)
        np.savetxt(ssim_file, ssim_gapfastdvdnet)
    
    if SAVE_MEAS:
        if not os.path.exists(savedmatdir + 'meas/'):
            os.makedirs(savedmatdir + 'meas/')
        for i in range(meas.shape[2]):
            plt.imsave('{}meas/meas{}.jpeg'.format(savedmatdir, i), meas[:,:,i], cmap='Greys_r')
    