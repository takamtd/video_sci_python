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

# from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version


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

class UNROLL_PNP_FFDNET(nn.Module): 
    def __init__(self, max_layers, denoiser):
        super(UNROLL_PNP_FFDNET, self).__init__()
        if denoiser == 'my_tv':
            self.tv_weight = nn.Parameter(1.*torch.ones(max_layers))
        else:
            self.sigma = nn.Parameter(50/255*torch.ones(max_layers))
        print("UNROLL_PNP_FFDNET initialized...")

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
                        tv_initialize = None, 
                        projmeth=None, v0=None, orig=None,
                        MAXB=255, maskdirection='plain',
                        denoiser=None, model=None, _lambda=None,
                        accelerate=None):
        self.mask = mask
        if projmeth.lower() == 'gap':
            if tv_initialize:
                if denoiser == 'my_tv':
                    vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                        tv_weight = self.tv_weight,
                                                        projmeth=projmeth, v0=vgaptv, orig=orig,
                                                        _lambda=_lambda,
                                                        MAXB=MAXB, maskdirection='plain',
                                                        denoiser=denoiser, model=model, 
                                                        iter_max=layer_num)
                else:
                    vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                        tv_weight = 0.1,
                                                        projmeth=projmeth, v0=vgaptv, orig=orig,
                                                        _lambda=_lambda,
                                                        MAXB=MAXB, maskdirection='plain',
                                                        denoiser=denoiser, model=model, 
                                                        iter_max=layer_num, sigma=self.sigma)
            else:
                if denoiser == 'my_tv':
                    vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                        tv_weight = self.tv_weight,
                                                        projmeth=projmeth, v0=None, orig=orig,
                                                        MAXB=MAXB, maskdirection='plain',
                                                        _lambda=_lambda, accelerate=accelerate,
                                                        denoiser=denoiser, model=model, 
                                                        iter_max=layer_num)
                else:
                    vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                        tv_weight = 0.1,
                                                        projmeth=projmeth, v0=None, orig=orig,
                                                        MAXB=MAXB, maskdirection='plain',
                                                        _lambda=_lambda, accelerate=accelerate,
                                                        denoiser=denoiser, model=model, 
                                                        iter_max=layer_num, sigma=self.sigma)
        elif projmeth.lower() == 'admm':
            if tv_initialize:
                vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                    projmeth=projmeth, v0=vgaptv, orig=orig,
                                                    iframe=iframe,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=layer_num, sigma=self.sigma)
            else:
                vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, self.A, self.At,
                                                    projmeth=projmeth, v0=None, orig=orig,
                                                    iframe=iframe,
                                                    MAXB=MAXB, maskdirection='plain',
                                                    _lambda=_lambda,
                                                    denoiser=denoiser, model=model, 
                                                    iter_max=layer_num, sigma=self.sigma)
        else:
            print('Unsupported projection method %s' % projmeth.upper())
        print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}.'.format(projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet)))
        return vgapffdnet


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
alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
allnframes = [      -1,         -1,         1,       1,       -1,        -1]
nframe = -1
# alldatname = ['kobe32']
# allnframes = [      -1]


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
    
    for meas_num in range(meas.shape[2]):
        data = []
        data.append(meas[:,:,meas_num])
        data.append(mask)
        start_point = meas_num*mask.shape[2]
        data.append(orig[:, :, start_point : start_point + mask.shape[2]])
        alldata.append(data)

data_num = len(alldata)

iframe = 0
# nframe = 1
# nframe = meas.shape[2]
if nframe < 0:
    nframe = meas.shape[2]
MAXB = 255.

# common parameters and pre-calculation for PnP
# define forward model and its transpose
# A  = lambda x :  A_(x, mask) # forward model function handle
# At = lambda y : At_(y, mask) # transpose of forward model

# mask_sum = np.sum(mask, axis=2)
# mask_sum[mask_sum==0] = 1


# Parameter setting
# max_layers = 80
max_layers = 2
adam_lr = 0.04  # initial learning parameter for Adam
batch_num = 7
# epoch_num = 280
epoch_num = 10

# In[6]:
#################################################################################
projmeth = 'gap'
tv_initialize = False

## [2.5] GAP/ADMM-FFDNet
### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
# projmeth = 'gap' # projection method
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'my_tv' 
# denoiser = 'ffdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP
# sigma    = [100/255, 50/255, 25/255, 12/255] # pre-set noise standard deviation
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
network = UNROLL_PNP_FFDNET(max_layers, denoiser).to(device)  # generating an instance of TISTA network
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)
train, test = separate_train_test(alldata, 1/7)
start = time.time()

mseloss = torch.nn.MSELoss()

# training process  
for epoch in range(epoch_num):
    print()
    print('epoch{}'.format(epoch+1))
    print()
    for i in range(int(len(train)/batch_num)):
        opt.zero_grad()
        meas_t_np, mask_t_np, orig_t_np = generate_batch(train, batch_num)
        # print(meas_t_np[0].shape)
        for batch in range(batch_num):
            # if (layer > 10): # change learning rate of Adam
            #     opt = optim.Adam(network.parameters(), lr=adam_lr/50.0)
            meas_t = torch.Tensor(meas_t_np[batch]).to(device)
            mask_t = torch.Tensor(mask_t_np[batch]).to(device)
            orig_t = torch.Tensor(orig_t_np[batch]).to(device)
            # meas_t[batch] = torch.Tensor(meas_t_np[batch]).to(device)
            # mask_t[batch] = torch.Tensor(mask_t_np[batch]).to(device)
            # orig_t[batch] = torch.Tensor(orig_t_np[batch]).to(device)
            vgapffdnet = network(meas_t, mask_t, max_layers,
                                projmeth=projmeth, v0=None, orig=orig_t,
                                MAXB=MAXB, maskdirection='plain',
                                _lambda=_lambda, accelerate=accelerate,
                                denoiser=denoiser, model=model).to(device)
            # x_hat = network(x, s_zero, layer+1).to(device)
            # print(vgapffdnet.shape)
        
            # loss = network.loss(vgapffdnet, orig_t)
            # print(meas_t[:,:])
            # a = torch.Tensor([0.1,0.2,0.3]).to(device)
            loss = mseloss(vgapffdnet, orig_t)
            for param in network.parameters():
                print("grad:{}".format(param.grad))
            loss.backward()
            print("backward")
            # print(vgapffdnet)

        # grads = torch.stack([param.grad for param in network.parameters()])
        # if isnan(grads).any():  # avoiding NaN in gradients
        #     continue
        opt.step()
        print()
        print("update")
        print()
        print("Parameter:{}".format(list(network.parameters()) ))
        # print()
        # print()
        # print("PSNR:{}", 20 * torch.log10(1.0 / torch.sqrt(loss)))
        # print("PSNR:{}", loss)

# incremental trainnig
# for layer in range(max_layers):
#     # training process  
#     for epoch in range(epoch_num):
#         print()
#         print(layer+1)
#         print('epoch{}'.format(epoch+1))
#         print()
#         for i in range(int(len(train)/batch_num)):
#             opt.zero_grad()
#             meas_t_np, mask_t_np, orig_t_np = generate_batch(train, batch_num)
#             # print(meas_t_np[0].shape)
#             for batch in range(batch_num):
#                 # if (layer > 10): # change learning rate of Adam
#                 #     opt = optim.Adam(network.parameters(), lr=adam_lr/50.0)
#                 meas_t = torch.Tensor(meas_t_np[batch]).to(device)
#                 mask_t = torch.Tensor(mask_t_np[batch]).to(device)
#                 orig_t = torch.Tensor(orig_t_np[batch]).to(device)
#                 # meas_t[batch] = torch.Tensor(meas_t_np[batch]).to(device)
#                 # mask_t[batch] = torch.Tensor(mask_t_np[batch]).to(device)
#                 # orig_t[batch] = torch.Tensor(orig_t_np[batch]).to(device)
#                 vgapffdnet = network(meas_t, mask_t, layer+1,
#                                     projmeth=projmeth, v0=None, orig=orig_t,
#                                     MAXB=MAXB, maskdirection='plain',
#                                     _lambda=_lambda, accelerate=accelerate,
#                                     denoiser=denoiser, model=model).to(device)
#                 # x_hat = network(x, s_zero, layer+1).to(device)
#                 # print(vgapffdnet.shape)
            
#                 # loss = network.loss(vgapffdnet, orig_t)
#                 # print(meas_t[:,:])
#                 # a = torch.Tensor([0.1,0.2,0.3]).to(device)
#                 loss = mseloss(vgapffdnet, orig_t)
#                 for param in network.parameters():
#                     print("grad:{}".format(param.grad))
#                 loss.backward()
#                 print("backward")
#                 # print(vgapffdnet)

#             # grads = torch.stack([param.grad for param in network.parameters()])
#             # if isnan(grads).any():  # avoiding NaN in gradients
#             #     continue
#             opt.step()
#             print()
#             print("update")
#             print()
#             print("Parameter:{}".format(list(network.parameters()) ))
#             # print()
#             # print()
#             # print("PSNR:{}", 20 * torch.log10(1.0 / torch.sqrt(loss)))
#             # print("PSNR:{}", loss)

#     # accuracy check after t-th incremental training
#     # loss_sum = 0.0
#     # for i in range(len(test)):
#     #     meas = test[i][0]
#     #     mask = test[i][1]
#     #     orig = test[i][2]

#     #     meas = torch.Tensor(meas).to(device)
#     #     mask = torch.Tensor(mask).to(device)
#     #     orig = torch.Tensor(orig).to(device)
#     #     vgapffdnet = network(meas, mask, layer,
#     #                         projmeth=projmeth, v0=None, orig=orig,
#     #                         MAXB=MAXB, maskdirection='plain',
#     #                         _lambda=_lambda, accelerate=accelerate,
#     #                         denoiser=denoiser, model=model).to(device)
#     #     loss_sum += psnr_gapffdnet

#     # loss_mean = loss_sum / len(test)

#     # print('({0}) PSNR= {}'.format(layer + 1, nmse))

# from torch.utils.tensorboard import SummaryWriter



# writer = SummaryWriter('experiment/test_experiment_1')

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
 
# # create grid of images
# img_grid = torchvision.utils.make_grid(images)
 
# # show images
# matplotlib_imshow(img_grid, one_channel=True)
 
# # write to tensorboard
# writer.add_image('four_fashion_mnist_images', img_grid)



# img = make_dot(vgapffdnet, params=network.parameters())
# plt.imsave(('c_graph'), img)

# end of training training

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

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

