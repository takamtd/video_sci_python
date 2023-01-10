import os
import time
import math
import h5py
import csv
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
from packages.fastdvdnet.models import FastDVDnet

# from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def separate_train_test(dataset, test_rate):
    # test_rate 0.0~1.0
    np.random.seed(seed=0)
    num_all = len(dataset)
    id_all = np.random.choice(num_all, num_all, replace=False)
    num_test = int(num_all*test_rate)
    id_test  = id_all[0:num_test]
    id_train = id_all[num_test:num_all]
    test_dataset = []
    train_dataset = []
    for id in id_test:
        test_dataset.append(dataset[id])
    for id in id_train:
        train_dataset.append(dataset[id])
    return train_dataset, test_dataset

def generate_batch(dataset, batch_num):
    meas = []
    mask = []
    orig = []
    ids = np.random.choice(len(dataset), batch_num, replace=False)
    for id in ids:
        meas.append(dataset[id][0])
        mask.append(dataset[id][1])
        orig.append(dataset[id][2])
    return meas, mask, orig

def generate_batch2(dataset, batch_num, start):
    meas = []
    mask = []
    orig = []
    ids = np.array([start+i for i in range(batch_num)])
    for id in ids:
        meas.append(dataset[id][0])
        mask.append(dataset[id][1])
        orig.append(dataset[id][2])
    return meas, mask, orig

class UNROLL_PNP_FASTDVDNET(nn.Module): 
    def __init__(self, max_layers, train_delta=True, train_gamma=False, sigmoid=True):
        super(UNROLL_PNP_FASTDVDNET, self).__init__() 
        self.sigmoid = sigmoid
        # self.sigma = nn.Parameter(100/255*torch.ones(max_layers))
        self.sigma = nn.Parameter(torch.cat([-1.410986973710262*torch.ones(20), -2.2192034840549946*torch.ones(20) ,-3.008154793552548*torch.ones(20)]))
        # self.sigma = nn.Parameter(torch.cat([-0.4382549309311554*torch.ones(20), -1.410986973710262*torch.ones(20), -2.2192034840549946*torch.ones(20) ,-3.008154793552548*torch.ones(20)]))
        # self.sigma = nn.Parameter(-3.008154793552548*torch.ones(max_layers))
        # self.sigma = nn.Parameter(-0.4382549309311555*torch.ones(max_layers))
        self.train_delta = train_delta
        self.train_gamma = train_gamma
        if train_delta:
            self.delta = nn.Parameter(1.0*torch.ones(max_layers))
        if train_gamma:
            self.gamma = nn.Parameter(0.01*torch.ones(max_layers))
        print("UNROLL_PNP_FASTDVDNET initialized...")

    def A(self, x):
        '''
        Forward model of snapshot compressive imaging (SCI), where multiple coded
        frames are collapsed into a snapshot measurement.
        '''
        # # for 3-D measurements only
        # return np.sum(x*Phi, axis=2)  # element-wise product
        # for general N-D measurements
        return torch.sum(x*self.mask, axis=tuple(range(2,self.mask.ndim)))  # element-wise product
    
    def At(self, y):
        '''
        Tanspose of the forward model. 
        '''
        # print(self.mask.permute(2,1,0).shape)
        # print(y.shape)
        # (nrow, ncol, nmask) = Phi.shape
        # x = np.zeros((nrow, ncol, nmask))
        # for nt in range(nmask):
        #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
        # return x
        # # for 3-D measurements only
        # return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)
        # for general N-D measurements (original Phi: H x W (x C) x F, y: H x W)
        # [expected] direct broadcasting (Phi: F x C x H x W, y: H x W)
        # [todo] change the dimension order to follow NumPy convention
        # D = Phi.ndim
        # ax = tuple(range(2,D))
        # return np.multiply(np.repeat(np.expand_dims(y,axis=ax),Phi.shape[2:D],axis=ax), Phi) # inefficient considering the memory layout https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
        # return torch.permute(torch.permute(self.mask)*torch.permute(y))
        # print(self.mask.permute(2,1,0).shape)
        # print(y.transpose(1,0).shape)
        # print((self.mask.permute(2,1,0)*y.transpose(1,0)).permute(2,1,0).shape)
        return (self.mask.permute(2,1,0)*y.transpose(1,0)).permute(2,1,0)

    def loss(self, x, orig):
        '''
        outputs: 予測結果
        targets: 正解
        '''
        # 損失の計算
        loss = 0
        for i in range(orig.shape[2]):
            loss += torch.mean((orig[:,:,i] - x[:,:,i]) ** 2)
        return loss/orig.shape[2]

    def forward(self,   meas, mask, layer_num,
                        tv_initialize = False, 
                        projmeth=None, v0=None, orig=None,
                        MAXB=255., maskdirection='plain',
                        denoiser=None, model=None, _lambda=None, sigmoid=False,
                        accelerate=None):
        self.mask = mask
        if projmeth.lower() == 'gap':
            if tv_initialize:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=vgaptv, orig=orig,
                                                    _lambda=_lambda,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
            elif self.train_delta:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,  delta=torch.relu(self.delta),
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
            else:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda, accelerate=accelerate,
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
        elif projmeth.lower() == 'admm':
            if tv_initialize:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                    projmeth=projmeth, v0=vgaptv, orig=orig,
                                                    iframe=iframe,
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
            elif self.train_gamma:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,  gamma=torch.relu(self.gamma),
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
            else:
                vgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet,psnrall_gapfastdvdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                    denoiser=denoiser, model=model, sigmoid=self.sigmoid,
                                                    iter_max=layer_num, sigma=self.sigma)
        else:
            print('Unsupported projection method %s' % projmeth.upper())
        print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}.'.format(projmeth.upper(), denoiser.upper(), mean(psnr_gapfastdvdnet), mean(ssim_gapfastdvdnet)))
        # print(psnr_gapfastdvdnet, ssim_gapfastdvdnet)
        return vgapfastdvdnet.to('cuda')


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

# 全データ数　28
# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1,        -1]
            #  [       4,          6,         5,       5,        4,         4]
            #  [     0~3,        4~9,     10~14,   11~19,    20~23,     24~27]
nframe = -1
# alldatname = ['kobe32']
# allnframes = [      -1]
# alldatname = ['kobe32','runner40','drop40','crash32','aerial32']
# allnframes = [      -1,         -1,         1,       1,       -1]
alldatname = ['traffic48','runner40','drop40','crash32','aerial32']
allnframes = [        -1,         -1,         1,       1,       -1]

useGPU = True # use GPU

if useGPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load data
alldata = []
for datname, nframe in zip(alldatname, allnframes):
    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

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
    print(meas)
    meas = torch.from_numpy(meas).to(device)
    mask = torch.from_numpy(mask).to(device)
    orig = torch.from_numpy(orig).to(device)
    for meas_num in range(meas.shape[2]):
        data = []
        data.append(meas[:,:,meas_num])
        data.append(mask)
        start_point = meas_num*mask.shape[2]
        data.append(orig[:, :, start_point : start_point + mask.shape[2]])
        alldata.append(data)

data_num = len(alldata)
# print(data_num)

iframe = 0
# nframe = 1
# nframe = meas.shape[2]
if nframe < 0:
    nframe = meas.shape[2]
MAXB = 255.


########################################################################
import torch
from packages.fastdvdnet.my_models import FastDVDnet

## [2.2] GAP-FastDVDnet
projmeth = 'gap' # projection method
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'fastdvdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP
train_delta = False
train_gamma = False

resultsdir = "./results/trainning_data/" + projmeth + '/'
file_n = '_' + "ex_kobe_method1_acc_lr0005"
# file_n = '_' + "ex_kobe_method1_acc_delta"

# Parameter setting
max_layers = 60
adam_lr = 0.005  # initial learning parameter for Adam
batch_num = 7
epoch_num = 280
# batch_num = 1
# epoch_num = 140
# epoch_num = 1

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


# trainning parameters
network = UNROLL_PNP_FASTDVDNET(max_layers, train_delta=train_delta, train_gamma=train_gamma).to(device)  # generating an instance of TISTA network
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)
# train, test = separate_train_test(alldata, 0)
train = alldata
# test = torch.Tensor(test).to(device)

start = time.time()

mseloss = torch.nn.MSELoss()
max_layers = 1
# incremental trainnig
for layer in range(max_layers):
    # training process
    # if layer == 0:
    #     layer = 29
    # else:
    #     layer = 59
    layer = 59
    count = 0
    for epoch in range(epoch_num):
        print(layer+1)
        print('epoch{}'.format(epoch+1))
        print()
        psnr_ep_sum = 0
        ssim_ep_sum = 0
        loss_ep_sum = 0
        for i in range(int(len(train)/batch_num)):
            opt.zero_grad()
            # meas_t_np, mask_t_np, orig_t_np = generate_batch(train, batch_num)
            meas_t, mask_t, orig_t = generate_batch(train, batch_num)
            # meas_t, mask_t, orig_t = generate_batch2(train, batch_num, count)
            # count += batch_num
            # count %= len(train)
            loss_sum = 0.
            psnr_sum = 0.
            ssim_sum = 0.
            for batch in range(batch_num):
                psnr_tb = 0.
                ssim_tb = 0.
                meas_tb = meas_t[batch]
                mask_tb = mask_t[batch]
                orig_tb = orig_t[batch]
                vgapfastdvdnet = network(meas_tb, mask_tb, layer+1,
                                    projmeth=projmeth, v0=None, orig=orig_tb,
                                    MAXB=MAXB, maskdirection='plain',
                                    _lambda=_lambda, accelerate=accelerate,
                                    denoiser=denoiser, model=model)
                
                orig_tb = orig_tb / MAXB
                loss = mseloss(vgapfastdvdnet, orig_tb)
                loss.backward()
                print("backward")
                print("loss:{}".format(loss))
                for param in network.parameters():
                    print("grad:{}".format(param.grad))
                print()
                vgapfastdvdnet_np = vgapfastdvdnet.to('cpu').detach().numpy().copy()
                orig_tb_np = orig_tb.to('cpu').detach().numpy().copy()
                nmask = vgapfastdvdnet_np.shape[-1]
                for imask in range(nmask):
                    psnr_tb += peak_signal_noise_ratio(orig_tb_np[...,imask], vgapfastdvdnet_np[...,imask], data_range=1.)
                    ssim_tb += structural_similarity(orig_tb_np[...,imask], vgapfastdvdnet_np[...,imask], data_range=1.,multichannel=vgapfastdvdnet_np[...,imask].ndim>2)
                psnr_sum += psnr_tb / nmask
                ssim_sum += ssim_tb / nmask
                print(psnr_tb / nmask, ssim_tb / nmask)
                loss_sum += loss
            
            psnr_sum /= batch_num
            ssim_sum /= batch_num
            loss_sum /= batch_num

            # psnr_ep_sum += psnr_sum
            # ssim_ep_sum += ssim_sum
            # loss_ep_sum += loss_sum

            opt.step()

            print()
            print("update!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Parameter:{}".format(list(network.parameters()) ))
            print()
            print()

            with open(resultsdir + 'data_files/loss{}.csv'.format(file_n), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, loss_sum])
            
            with open(resultsdir + 'data_files/psnr{}.csv'.format(file_n), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, psnr_sum])
            with open(resultsdir + 'data_files/ssim{}.csv'.format(file_n), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, ssim_sum])
            with open(resultsdir + 'data_files/sigma{}.csv'.format(file_n), 'a') as f:
                sigma_list = []
                for sigma in network.state_dict()['sigma']:
                    sigma_list.append(float(sigma))
                writer = csv.writer(f)
                writer.writerow([epoch+1, sigma_list])
            if train_delta == True:
                with open(resultsdir + 'data_files/delta{}.csv'.format(file_n), 'a') as f:
                    delta_list = []
                    for delta in network.state_dict()['delta']:
                        delta_list.append(float(delta))
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, delta_list])
            if train_gamma == True:
                with open(resultsdir + 'data_files/gamma{}.csv'.format(file_n), 'a') as f:
                    gamma_list = []
                    for gamma in network.state_dict()['gamma']:
                        gamma_list.append(float(gamma))
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, gamma_list])
        
        # psnr_ep_sum /= int(len(train)/batch_num)
        # ssim_ep_sum /= int(len(train)/batch_num)
        # loss_ep_sum /= int(len(train)/batch_num)
        # with open(resultsdir + 'data_files/loss{}.csv'.format(file_n), 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch+1, loss_ep_sum])
        # with open(resultsdir + 'data_files/param{}.csv'.format(file_n), 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch+1, list(network.parameters())])
        # with open(resultsdir + 'data_files/psnr{}.csv'.format(file_n), 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch+1, psnr_ep_sum])
        # with open(resultsdir + 'data_files/ssim{}.csv'.format(file_n), 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch+1, ssim_ep_sum])
        

    # accuracy check after t-th incremental training
    # loss_sum = 0.0
    # psnr_sum = 0.0
    # ssim_sum = 0.0
    # for i in range(len(test)):
    #     meas = test[i][0]
    #     mask = test[i][1]
    #     orig = test[i][2]

    #     vgapfastdvdnet = network(meas, mask, layer,
    #                         projmeth=projmeth, v0=None, orig=orig,
    #                         MAXB=MAXB, maskdirection='plain',
    #                         _lambda=_lambda, accelerate=accelerate,
    #                         denoiser=denoiser, model=model).to(device)
    #     vgapfastdvdnet_np = vgapfastdvdnet.to('cpu').detach().numpy().copy()
    #     orig_np = orig.to('cpu').detach().numpy().copy() / MAXB
    #     psnr_sum += peak_signal_noise_ratio(orig_np, vgapfastdvdnet_np, data_range=1.)
    #     ssim_sum += structural_similarity(orig_np, vgapfastdvdnet_np, data_range=1.,multichannel=vgapfastdvdnet_np.ndim>2)
    #     loss_sum += mseloss(vgapfastdvdnet, orig/MAXB)

    # if len(test) != 0:
    #     psnr_mean = psnr_sum / len(test)
    #     ssim_mean = ssim_sum / len(test)
    #     loss_mean = loss_sum / len(test)

    #     print('({0}) PSNR= {1}'.format(layer + 1, psnr_mean))
    #     print('({0}) SSIM= {1}'.format(layer + 1, ssim_mean))
    #     print('({0}) MSE= {1}'.format(layer + 1, loss_mean))
    #     with open(resultsdir + 'data_files/result{}.csv'.format(file_n), 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([i+1, psnr_mean, ssim_mean, loss_mean])

# end of training training

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
