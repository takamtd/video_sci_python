import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from my_pnp_sci import admmdenoise_cacti

from utils import (A_, At_)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from packages.ffdnet.models import FFDNet

from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version

def generate_test_train():
    
    return test, train

def generate_batch():
    return 0

def inverse_psnr(psnr, MAXB=255):
    return MAXB**2/10**(psnr/10)

def CustomLoss(psnr):
    '''
    outputs: 予測結果
    targets: 正解
    '''
    # 損失の計算
    loss = inverse_psnr(psnr)
    return loss

class UNROLL_PNP_FFDNET(nn.Module): 
    def __init__(self, max_layers):
        super(UNROLL_PNP_FFDNET, self).__init__() 
        self.sigma = nn.Parameter(1/255*torch.ones(max_layers))
        print("UNROLL_PNP_FFDNET initialized...")
        
    def forward(self,   meas, mask, A, At, 
                        tv_initialize = None, 
                        projmeth=None, v0=None, orig=None,
                        iframe=None, nframe=None,
                        MAXB=255, maskdirection='plain',
                        _lambda=None, accelerate=None,
                        denoiser=None, model=None, 
                        iter_max=[1 for i in range(80)]):
        
        if projmeth.lower() == 'gap':
            if tv_initialize:
                vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=vgaptv, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=self.sigma)
            else:
                vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=self.sigma)
        elif projmeth.lower() == 'admm':
            if tv_initialize:
                vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                  MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                      projmeth=projmeth, v0=vgaptv, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=self.sigma)
            else:
                vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe, nframe=nframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=iter_max, sigma=self.sigma)
        else:
            print('Unsupported projection method %s' % projmeth.upper())
            
        print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet))
        return vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet


# datname = 'kobe32'        # name of the dataset
# datname = 'traffic48'     # name of the dataset
# datname = 'runner40'      # name of the dataset
# datname = 'drop40'        # name of the dataset
# datname = 'crash32'       # name of the dataset
# datname = 'aerial32'      # name of the dataset
# datname = 'bicycle24'     # name of the dataset
# datname = 'starfish48'    # name of the dataset

# datname = 'starfish_c16_48'    # name of the dataset

# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
allnframes = [      -1,         -1,         1,       1,       -1,        -1]
nframe = -1
# alldatname = ['kobe32']
# allnframes = [      -1]

# load data
alldata = []
for datname, nframe in zip(alldatname, allnframes):
    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
        meas = np.float32(file['meas'])
        mask = np.float32(file['mask'])
        orig = np.float32(file['orig'])
    else: # MATLAB .mat v7.3
        file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        meas = np.float32(file['meas']).transpose()
        mask = np.float32(file['mask']).transpose()
        orig = np.float32(file['orig']).transpose()
    
    data = []
    for meas_num in range(meas.shape[2]):
        data.append(meas[meas_num])
        data.append(mask)
        start_point = meas_num*mask.shape[2]
        data.append(orig[start_point : start_point + mask.shape[2] - 1])
    
    alldata.append(data)

iframe = 0
# nframe = 1
# nframe = meas.shape[2]
if nframe < 0:
    nframe = meas.shape[2]
MAXB = 255.
print(orig.shape[2])

# common parameters and pre-calculation for PnP
# define forward model and its transpose
A  = lambda x :  A_(x, mask) # forward model function handle
At = lambda y : At_(y, mask) # transpose of forward model

mask_sum = np.sum(mask, axis=2)
mask_sum[mask_sum==0] = 1

max_layers = 80
adam_lr = 0.04  # initial learning parameter for Adam
batch_num = 200

# In[6]:
#################################################################################
projmeth = 'gap'
method_type = 8
tv_initialize = False

## [2.5] GAP/ADMM-FFDNet
### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
# projmeth = 'gap' # projection method
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'ffdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP
# sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
iter_max = [1 for i in range(80)] # maximum number of iterations
useGPU = True # use GPU

# pre-load the model for FFDNet image denoising
in_ch = 1
model_fn = 'packages/ffdnet/models/net_gray.pth'
# Absolute path to model file
# model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

# Create model
net = FFDNet(num_input_channels=in_ch)
# Load saved weights
if useGPU:
    device = torch.device('cuda')
    state_dict = torch.load(model_fn)
    device_ids = [0]
    model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
else:
    state_dict = torch.load(model_fn, map_location='cpu')
    # CPU mode: remove the DataParallel wrapper
    state_dict = remove_dataparallel_wrapper(state_dict)
    model = net
model.load_state_dict(state_dict)
model.eval() # evaluation mode

# trainning parameters
network = UNROLL_PNP_FFDNET(max_layers).to(device)  # generating an instance of TISTA network
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)

start = time.time()

# incremental trainnig
for layer in (range(max_layers)):
    # training process  
    for i in range(batch_num):
        if (layer > 10): # change learning rate of Adam
            opt = optim.Adam(network.parameters(), lr=adam_lr/50.0)
        meas_t, mask_t, orig_t = torch.Tensor(generate_batch()).to(device)
        opt.zero_grad()
        vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = network(meas, mask, A, At,
                                                                                projmeth=projmeth, v0=None, orig=orig,
                                                                                iframe=iframe, nframe=nframe,
                                                                                MAXB=MAXB, maskdirection='plain',
                                                                                _lambda=_lambda, accelerate=accelerate,
                                                                                denoiser=denoiser, model=model, 
                                                                                iter_max=iter_max).to(device)
        # x_hat = network(x, s_zero, layer+1).to(device)
        loss = CustomLoss(mean(psnr_gapffdnet))
        loss.backward()

        grads = torch.stack([param.grad for param in network.parameters()])
        if isnan(grads).any():  # avoiding NaN in gradients
            continue

        opt.step()
        print("PSNR:{}", mean(psnr_gapffdnet))
# end of training training

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

accuracy check after t-th incremental training
nmse_sum = 0.0
tot = 1 # batch size for accuracy check
for i in range(tot):
    x = torch.Tensor(generate_batch()).to(device)
    x_hat = network(x, s_zero, gen+1).to(device)
    num = (x - x_hat).norm(2, 1).pow(2.0)
    denom = x.norm(2,1).pow(2.0)
    nmse = num/denom
    nmse_sum += torch.sum(nmse).item()

nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]

print('({0}) NMSE= {1:6.3f}'.format(gen + 1, nmse))


###########################################

# In[8]:
# [3.3] result demonstration of GAP-Denoise
# nmask = mask.shape[2]

# SAVE_RESULT = True
# SAVE_DATA = True
# SAVE_MEAS = False

# savedmatdir = resultsdir + '/savedmat/grayscale/' + alldatname[0] + '/'
# if not os.path.exists(savedmatdir):
#     os.makedirs(savedmatdir)

# psnrall_gapffdnet = np.array(psnrall_gapffdnet)
# psnrmean_gapffdnet = psnrall_gapffdnet.mean(axis=0)
# sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
#             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
#
# if SAVE_RESULT:
#     if not os.path.exists(savedmatdir + 'gaptv/'):
#         os.makedirs(savedmatdir + 'gaptv/')
#     if not os.path.exists(savedmatdir + projmeth + 'ffdnet/'):
#         os.makedirs(savedmatdir + projmeth + 'ffdnet/')
#     if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/'):
#         os.makedirs(savedmatdir + projmeth + 'fastdvdnet/')
#     if tv_initialize:
#         if not os.path.exists(savedmatdir + projmeth + 'ffdnet/method{:d}_tv_initialize/'.format(method_type)):
#             os.makedirs(savedmatdir + projmeth + 'ffdnet/method{:d}_tv_initialize/'.format(method_type))
#         if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/method{:d}_tv_initialize/'.format(method_type)):
#             os.makedirs(savedmatdir + projmeth + 'fastdvdnet/method{:d}_tv_initialize/'.format(method_type))
#     else:
#         if not os.path.exists(savedmatdir + 'gaptv/method{:d}/'.format(method_type)):
#             os.makedirs(savedmatdir + 'gaptv/method{:d}/'.format(method_type))
#         if not os.path.exists(savedmatdir + projmeth + 'ffdnet/method{:d}/'.format(method_type)):
#             os.makedirs(savedmatdir + projmeth + 'ffdnet/method{:d}/'.format(method_type))
#         if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/method{:d}/'.format(method_type)):
#             os.makedirs(savedmatdir + projmeth + 'fastdvdnet/method{:d}/'.format(method_type))
#     for i in range(orig.shape[2]):
#         if tv_initialize:
#             iter_max = 5
#             if i < 10:
#                 plt.imsave('{}{projmeth}ffdnet/method{:d}_tv_initialize/{}_{projmeth}ffdnet_tv_initialize{:d}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}fastdvdnet/method{:d}_tv_initialize/{}_{projmeth}fastdvdnet_tv_initialize{:d}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
#             else:
#                 plt.imsave('{}{projmeth}ffdnet/method{:d}_tv_initialize/{}_{projmeth}ffdnet_tv_initialize{:d}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}fastdvdnet/method{:d}_tv_initialize/{}_{projmeth}fastdvdnet_tv_initialize{:d}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
#         else:
#             if i < 10:
#                 plt.imsave('{}gaptv/method{:d}/{}_gaptv_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i), vgaptv[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}ffdnet/method{:d}/{}_{projmeth}ffdnet_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}fastdvdnet/method{:d}/{}_{projmeth}fastdvdnet_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
#             else:
#                 plt.imsave('{}gaptv/method{:d}/{}_gaptv_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i), vgaptv[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}ffdnet/method{:d}/{}_{projmeth}ffdnet_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
#                 plt.imsave('{}{projmeth}fastdvdnet/method{:d}/{}_{projmeth}fastdvdnet_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')

# if SAVE_DATA:
#     if not os.path.exists(savedmatdir + 'data/'):
#         os.makedirs(savedmatdir + 'data/')
#     if tv_initialize:
#         filelast_name = "_psnr_method{:d}_tv_initialize{:d}.csv".format(method_type, iter_max)
#     else:
#         filelast_name = "_psnr_method{:d}.csv".format(method_type)
#     psnrall_gaptv = np.array(psnrall_gaptv)
#     psnrall_gapffdnet = np.array(psnrall_gapffdnet)
#     psnrall_gapfastdvdnet = np.array(psnrall_gapfastdvdnet)
#     psnrmean_gaptv = psnrall_gaptv.mean(axis=0)
#     psnrmean_gapffdnet = psnrall_gapffdnet.mean(axis=0)
#     psnrmean_gapfastdvdnet = psnrall_gapfastdvdnet.mean(axis=0)
#     gaptv_path = savedmatdir + 'data/' + "gaptv" + filelast_name
#     ffdnet_path = savedmatdir + 'data/' + projmeth + "ffdnet" + filelast_name
#     fastdvdnet_path = savedmatdir + 'data/' + projmeth + "fastdvdnet" + filelast_name
#     np.savetxt(gaptv_path, psnrmean_gaptv)
#     np.savetxt(ffdnet_path, psnrmean_gapffdnet)
#     np.savetxt(fastdvdnet_path, psnrmean_gapfastdvdnet)

# if SAVE_MEAS:
#     if not os.path.exists(savedmatdir + 'meas/'):
#         os.makedirs(savedmatdir + 'meas/')
#     for i in range(meas.shape[2]):
#         plt.imsave('{}meas/meas{}.jpeg'.format(savedmatdir, i), meas[:,:,i], cmap='Greys_r')

