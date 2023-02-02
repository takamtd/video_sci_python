from PIL import Image
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.io.matlab.miobase import get_matfile_version

datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
# alldatname = ['traffic48','traffic48','runner40','drop40','crash32','aerial32']
datname = 'traffic48'
matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

resultsdir = './results'
savedatadir = resultsdir + '/savedmat/grayscale/'

# [1] load data
# if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
meas = np.float32(file['meas'])
mask = np.float32(file['mask'])
orig = np.float32(file['orig'])
# else: # MATLAB .mat v7.3
#     file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
#     meas = np.float32(file['meas']).transpose()
#     mask = np.float32(file['mask']).transpose()
#     orig = np.float32(file['orig']).transpose()

# print(orig[:,:,0])
 
# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    imgs = []                                                   # 画像をappendするための空配列を定義
    # print(path_list)
    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        imgs.append(img)                                        # 画像をappendで配列に格納していく
 
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=800, loop=0)

def create_gif_from_np(data, out_filename):
    imgs = []                                                   # 画像をappendするための空配列を定義
    for i in range(data.shape[2]):
        # Numpy配列をImageオブジェクトに変換してリストに追加
        imgs.append(Image.fromarray(np.uint8(data[:,:,i])).convert('P'))
 
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=1000/20, loop=0)
 
# GIFアニメーションを作成する関数を実行する
# create_gif(in_dir='/home/jovyan/workdir/results/savedmat/grayscale/kobe32/meas', out_filename='kobe_meas.gif')
create_gif_from_np(orig, out_filename=savedatadir+datname+'_orig.gif')
# create_gif_from_np(mask*255, out_filename='/home/jovyan/workdir/results/savedmat/grayscale/mask.gif')
