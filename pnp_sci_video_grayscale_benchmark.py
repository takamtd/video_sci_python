#!/usr/bin/env python
# coding: utf-8

# ## GAP-TV for Video Compressive Sensing
# ### GAP-TV
# > X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," in *IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 2539-2543.
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), y-liu16@mails.tsinghua.edu.cn, updated Jan 20, 2019.

# In[1]:


import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci import admmdenoise_cacti

from utils import (A_, At_)

# In[2]:


# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1,        -1]
# alldatname = ['drop40','crash32','aerial32']
# allnframes = [       1,       -1,        -1]
alldatname = ['traffic48']
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


    # In[3]:


    # from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version

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

    iframe = 0
    # nframe = 1
    # nframe = meas.shape[2]
    if nframe < 0:
        nframe = meas.shape[2]
    MAXB = 255.
    # print(orig.shape[2])

    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    A  = lambda x :  A_(x, mask) # forward model function handle
    At = lambda y : At_(y, mask) # transpose of forward model

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1

    # In[4]:
    ######################################################################################
    # [2.3] GAP/ADMM-TV
    ## [2.3.1] GAP-TV
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV)
    iter_max = 1 # maximum number of iterations
    tv_weight = 0.3 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 60 # TV denoising maximum number of iterations each

    vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                              projmeth=projmeth, v0=None, orig=orig,
                                              iframe=iframe, nframe=nframe,
                                              MAXB=MAXB, maskdirection='plain',
                                              _lambda=_lambda, accelerate=accelerate,
                                              denoiser=denoiser, iter_max=iter_max, 
                                              tv_weight=tv_weight, 
                                              tv_iter_max=tv_iter_max)

    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptv), mean(ssim_gaptv), tgaptv))


    # In[6]:
    #################################################################################
    projmeth = 'gap'
    # method_type = 9
    tv_initialize = False

    import torch
    from packages.ffdnet.models import FFDNet

    ## [2.5] GAP/ADMM-FFDNet
    ### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
    # projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'ffdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    method_type = 1
    if method_type == 1:
        # sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
        # iter_max = [20, 20, 20, 20] # maximum number of iterations
        sigma    = [50/255, 25/255, 12/255] # pre-set noise standard deviation
        iter_max = [20, 20, 20] # maximum number of iterations
    elif method_type == 2:
        sigma    = [100*0.97**i/255 for i in range(80)]
        iter_max = [1 for i in range(80)]
    elif method_type == 3:
        sigma    = [12/255]
        iter_max = [80]
    elif method_type == 4:
        sigma    = [(50*0.5**(i/30))/255 for i in range(80)]
        iter_max = [1 for i in range(80)]
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
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
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

    if projmeth.lower() == 'gap':
        if tv_initialize:
            vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=vgaptv, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
        else:
            vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
    elif projmeth.lower() == 'admm':
        if tv_initialize:
            vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=vgaptv, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
        else:
            vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=None, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda,
                                                denoiser=denoiser, model=model, 
                                                iter_max=iter_max, sigma=sigma)
    else:
        print('Unsupported projection method %s' % projmeth.upper())
        
    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet))


    # In[7]:
    ###########################################################################

    import torch
    from packages.fastdvdnet.models import FastDVDnet

    ## [2.2] GAP-FastDVDnet
    # projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'fastdvdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    method_type = 9
    if method_type == 1:
        sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
        iter_max = [20, 20, 20, 20] # maximum number of iterations
    elif method_type == 2:
        sigma    = [100*0.97**i/255 for i in range(80)]
        iter_max = [1 for i in range(80)]
    elif method_type == 3:
        sigma    = [12/255]
        iter_max = [80]
    elif method_type == 4:
        sigma    = [(50*0.5**(i/30))/255 for i in range(80)]
        iter_max = [1 for i in range(80)]
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
        # sigma   =   [0.3270,  0.3269,  0.3010,  0.2881,  0.2587,  0.2258,  0.1619,  0.1161,
        #             0.1148,  0.0830, -0.0247,  0.1036, -0.0521,  0.1235, -0.0294, -0.0169,
        #             -0.0012, -0.0263,  0.0936,  0.0562,  0.0585, -0.0095,  0.0908,  0.1010,
        #             0.0478,  0.0999,  0.0771,  0.0501,  0.0220,  0.0693, -0.0211,  0.0892,
        #             0.1347,  0.1110,  0.0388, -0.0235,  0.1038,  0.0728, -0.0289,  0.1295,
        #             0.0588, -0.0153, -0.0180,  0.1009, -0.0260,  0.0597,  0.0538, -0.0351,
        #             0.0165,  0.0458,  0.0516,  0.0678, -0.0415,  0.0716,  0.0179,  0.0158,
        #             -0.0128,  0.0138,  0.0104,  0.0775]
        iter_max = [1 for i in range(60)]
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

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

    if projmeth.lower() == 'gap':
        if tv_initialize:
            vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                                projmeth=projmeth, v0=vgaptv, orig=orig,
                                                iframe=iframe, nframe=nframe,
                                                MAXB=MAXB, maskdirection='plain',
                                                _lambda=_lambda, accelerate=accelerate,
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
    
    SAVE_RESULT = False
    SAVE_DATA = True
    SAVE_MEAS = False
    OPTION = True
    if OPTION:
        option_name = 'trained_test'

    savedmatdir = resultsdir + '/savedmat/grayscale/' + alldatname[0] + '/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)
    
    psnrall_gapffdnet = np.array(psnrall_gapffdnet)
    psnrmean_gapffdnet = psnrall_gapffdnet.mean(axis=0)
    # sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
    #             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
    if SAVE_RESULT:
        if not os.path.exists(savedmatdir + 'gaptv/'):
            os.makedirs(savedmatdir + 'gaptv/')
        if not os.path.exists(savedmatdir + projmeth + 'ffdnet/'):
            os.makedirs(savedmatdir + projmeth + 'ffdnet/')
        if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/'):
            os.makedirs(savedmatdir + projmeth + 'fastdvdnet/')
        
        if tv_initialize:
            if not os.path.exists(savedmatdir + projmeth + 'ffdnet/method{:d}_tv_initialize/'.format(method_type)):
                os.makedirs(savedmatdir + projmeth + 'ffdnet/method{:d}_tv_initialize/'.format(method_type))
            if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/method{:d}_tv_initialize/'.format(method_type)):
                os.makedirs(savedmatdir + projmeth + 'fastdvdnet/method{:d}_tv_initialize/'.format(method_type))
        elif OPTION:
            if not os.path.exists(savedmatdir + 'gaptv/method{:d}_{}/'.format(method_type, option_name)):
                os.makedirs(savedmatdir + 'gaptv/method{:d}_{}/'.format(method_type, option_name))
            if not os.path.exists(savedmatdir + projmeth + 'ffdnet/method{:d}_{}/'.format(method_type, option_name)):
                os.makedirs(savedmatdir + projmeth + 'ffdnet/method{:d}_{}/'.format(method_type, option_name))
            if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/method{:d}_{}/'.format(method_type, option_name)):
                os.makedirs(savedmatdir + projmeth + 'fastdvdnet/method{:d}_{}/'.format(method_type, option_name))
        else:
            if not os.path.exists(savedmatdir + 'gaptv/method{:d}/'.format(method_type)):
                os.makedirs(savedmatdir + 'gaptv/method{:d}/'.format(method_type))
            if not os.path.exists(savedmatdir + projmeth + 'ffdnet/method{:d}/'.format(method_type)):
                os.makedirs(savedmatdir + projmeth + 'ffdnet/method{:d}/'.format(method_type))
            if not os.path.exists(savedmatdir + projmeth + 'fastdvdnet/method{:d}/'.format(method_type)):
                os.makedirs(savedmatdir + projmeth + 'fastdvdnet/method{:d}/'.format(method_type))
        
        for i in range(orig.shape[2]):
            if tv_initialize:
                iter_max = 5
                if i < 10:
                    plt.imsave('{}{projmeth}ffdnet/method{:d}_tv_initialize/{}_{projmeth}ffdnet_tv_initialize{:d}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}_tv_initialize/{}_{projmeth}fastdvdnet_tv_initialize{:d}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
                else:
                    plt.imsave('{}{projmeth}ffdnet/method{:d}_tv_initialize/{}_{projmeth}ffdnet_tv_initialize{:d}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}_tv_initialize/{}_{projmeth}fastdvdnet_tv_initialize{:d}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], iter_max, i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
            elif OPTION:
                if i < 10:
                    plt.imsave('{}gaptv/method{:d}_{option_name}/{}_gaptv_{option_name}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, option_name=option_name), vgaptv[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}ffdnet/method{:d}_{option_name}/{}_{projmeth}ffdnet_{option_name}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth, option_name=option_name), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}_{option_name}/{}_{projmeth}fastdvdnet_{option_name}_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth, option_name=option_name), vgapfastdvdnet[:,:,i], cmap='Greys_r')
                else:
                    plt.imsave('{}gaptv/method{:d}_{option_name}/{}_gaptv_{option_name}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, option_name=option_name), vgaptv[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}ffdnet/method{:d}_{option_name}/{}_{projmeth}ffdnet_{option_name}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth, option_name=option_name), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}_{option_name}/{}_{projmeth}fastdvdnet_{option_name}_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth, option_name=option_name), vgapfastdvdnet[:,:,i], cmap='Greys_r')
            else:
                if i < 10:
                    plt.imsave('{}gaptv/method{:d}/{}_gaptv_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i), vgaptv[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}ffdnet/method{:d}/{}_{projmeth}ffdnet_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}/{}_{projmeth}fastdvdnet_0{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
                else:
                    plt.imsave('{}gaptv/method{:d}/{}_gaptv_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i), vgaptv[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}ffdnet/method{:d}/{}_{projmeth}ffdnet_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapffdnet[:,:,i], cmap='Greys_r')
                    plt.imsave('{}{projmeth}fastdvdnet/method{:d}/{}_{projmeth}fastdvdnet_{:d}.jpeg'.format(savedmatdir, method_type, alldatname[0], i, projmeth=projmeth), vgapfastdvdnet[:,:,i], cmap='Greys_r')
    
    if SAVE_DATA:
        if not os.path.exists(savedmatdir + 'data/'):
            os.makedirs(savedmatdir + 'data/')
        if tv_initialize:
            filelast_name = "_psnr_method{:d}_tv_initialize{:d}.csv".format(method_type, iter_max)
        elif OPTION:
            filelast_name = "_psnr_method{:d}_{}.csv".format(method_type, option_name)
        else:
            filelast_name = "_psnr_method{:d}.csv".format(method_type)
        psnrall_gaptv = np.array(psnrall_gaptv)
        psnrall_gapffdnet = np.array(psnrall_gapffdnet)
        psnrall_gapfastdvdnet = np.array(psnrall_gapfastdvdnet)
        psnrmean_gaptv = psnrall_gaptv.mean(axis=0)
        psnrmean_gapffdnet = psnrall_gapffdnet.mean(axis=0)
        psnrmean_gapfastdvdnet = psnrall_gapfastdvdnet.mean(axis=0)
        gaptv_path = savedmatdir + 'data/' + "gaptv" + filelast_name
        ffdnet_path = savedmatdir + 'data/' + projmeth + "ffdnet" + filelast_name
        fastdvdnet_path = savedmatdir + 'data/' + projmeth + "fastdvdnet" + filelast_name
        np.savetxt(gaptv_path, psnrmean_gaptv)
        np.savetxt(ffdnet_path, psnrmean_gapffdnet)
        np.savetxt(fastdvdnet_path, psnrmean_gapfastdvdnet)
    
    if SAVE_MEAS:
        if not os.path.exists(savedmatdir + 'meas/'):
            os.makedirs(savedmatdir + 'meas/')
        for i in range(meas.shape[2]):
            plt.imsave('{}meas/meas{}.jpeg'.format(savedmatdir, i), meas[:,:,i], cmap='Greys_r')
    
    # sio.savemat('{}gap{}_{}_{:d}_sigma{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
    #             {'vgaptv':vgaptv, 
    #              'psnr_gaptv':psnr_gaptv,
    #              'ssim_gaptv':ssim_gaptv,
    #              'psnrall_tv':psnrall_gaptv,
    #              'tgaptv':tgaptv,
    #              'vgapffdnet':vgapffdnet, 
    #              'psnr_gapffdnet':psnr_gapffdnet,
    #              'ssim_gapffdnet':ssim_gapffdnet,
    #              'psnrall_ffdnet':psnrall_gapffdnet,
    #              'tgapffdnet':tgapffdnet,
    #              'vgapfastdvdnet':vgapfastdvdnet, 
    #              'psnr_gapfastdvdnet':psnr_gapfastdvdnet,
    #              'ssim_gapfastdvdnet':ssim_gapfastdvdnet,
    #              'psnrall_fastdvdnet':psnrall_gapfastdvdnet,
    #              'tgapfastdvdnet':tgapfastdvdnet})

