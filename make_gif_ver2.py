from PIL import Image
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.io.matlab.miobase import get_matfile_version
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# datadir = '/home/jovyan/workdir/results/savedmat/grayscale/gap/ex_davis_method1_acc/'
datadir = '/home/jovyan/workdir/results/savedmat/grayscale/gap/method1_acc/'
dataname = 'traffic48'
savedatadir = datadir + dataname + '/'
inputdir = savedatadir + 'recon_imgs/'
savefile = savedatadir + dataname + '.gif'
# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
# datname = 'aerial32'
# matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file

# [1] load data
# if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
# file = sio.loadmat(matfile)
# meas = np.float32(file['meas'])
# mask = np.float32(file['mask'])
# orig = np.float32(file['orig'])

 
# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    imgs = []                                                   # 画像をappendするための空配列を定義
    # print(path_list)
    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        # img = Image.fromarray(np.array(img))
        imgs.append(img.convert('P'))                                        # 画像をappendで配列に格納していく
    print(len(imgs))
    print(imgs[-1])
    # print(img[0].mode)
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=1000/20, loop=0)

    # figオブジェクトを作る
    # fig = plt.figure()
    
    # ax = plt.subplot(1, 1, 1)
    # ax.spines['right'].set_color('None')
    # ax.spines['top'].set_color('None')
    # ax.spines['left'].set_color('None')
    # ax.spines['bottom'].set_color('None')
    # ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    # ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    # #空のリストを作る
    # imgs = []
    
    # #画像ファイルを順々に読み込んでいく
    # for i in range(len(path_list)):
    #     #1枚1枚のグラフを描き、appendしていく
    #     tmp = Image.open(path_list[i])
    #     imgs.append([plt.imshow(tmp)])
    
    # #アニメーション作成    
    # ani = animation.ArtistAnimation(fig, imgs, interval=800, repeat_delay=1000)
    # # ani = animation.ArtistAnimation(fig, imgs, interval=800, repeat_delay=1000)
    # ani.save(out_filename)

def create_gif_from_np(data, out_filename):
    imgs = []                                                   # 画像をappendするための空配列を定義
    for i in range(data.shape[2]):
        # Numpy配列をImageオブジェクトに変換してリストに追加
        imgs.append(Image.fromarray(np.uint8(data[:,:,i])).convert('P'))
 
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=100/8, loop=0)
 
# GIFアニメーションを作成する関数を実行する
create_gif(in_dir=inputdir, out_filename=savefile)
# create_gif_from_np(orig, path='/home/jovyan/workdir/results/savedmat/grayscale/'+datname+'/orig', out_filename=savedatadir+datname[:-2]+'_orig.gif')
# create_gif_from_np(mask*255, out_filename='/home/jovyan/workdir/results/savedmat/grayscale/mask.gif')
