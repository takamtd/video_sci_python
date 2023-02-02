import cv2
import os
import numpy as np
import scipy.io as sio

ROOT_PATH = '/home/jovyan/workdir/dataset/cacti/my_dataset/DAVIS/JPEGImages/480p'
OUT_PATH = '/home/jovyan/workdir/dataset/cacti/my_dataset/dataset'
MASK_PATH = '/home/jovyan/workdir/dataset/cacti/my_dataset/mask'

def crop_center(img, crop_width, crop_height):
    img_height, img_width = img.shape
    size = (crop_height * img_width // img_height, crop_height)
    img = cv2.resize(img, size)
    img_height, img_width = img.shape
    return img[(img_height - crop_height) // 2 : (img_height + crop_height) // 2,
               ( img_width -  crop_width) // 2 : ( img_width +  crop_width) // 2]

def process(file_path):
    # 処理を記述
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = crop_center(img, 256, 256)
    return img

def save_img(img, file_path):
    cv2.imwrite(file_path, img)

def add_noise(y_clear,sigma2):
    noise = np.random.normal(loc = 0, scale = y_clear.shape[2]*np.sqrt(sigma2), size = y_clear.shape) # sigmaでノイズ生成
    y = y_clear + noise # ブラー画像にノイズを不可
    return y

def get_mask():
    datasetdir = './dataset/cacti/grayscale_benchmark'
    datname = 'kobe32'
    matfile = datasetdir + '/' + datname + '_cacti.mat'
    file = sio.loadmat(matfile)
    mask = np.float32(file['mask'])
    return mask

def mask_img(origs, masks):
    imgs = []
    meas = np.zeros_like(masks[:,:,0])
    for i, orig in enumerate(origs):
        meas += (orig * masks[:,:,i])/len(origs)
    return meas

def recursive_file_check(path, save_path):
    # ディレクトリが１段の場合のみ有効
    np.random.seed(seed=0)
    count = 0
    B = 8
    masks = get_mask()
    dirnames = os.listdir(path)
    dirnames.sort()
    # print(dirnames)
    for dirname in dirnames:
        dirpath = path + "/" + dirname
        filenames = os.listdir(dirpath)
        filenames.sort()
        # print(filenames)
        for i in range(len(filenames)//B):
            imgs = []
            for j in range(B):
                img = process(dirpath + "/" + filenames[i*B + j])
                if not os.path.exists(save_path + "/" + dirname + "/"):
                    os.makedirs(save_path + "/" + dirname + "/")
                if not os.path.exists(save_path + "/" + dirname + "/" + "orig/"):
                    os.makedirs(save_path + "/" + dirname + "/" + "orig/")
                save_img(img, save_path + "/" + dirname + "/" + "orig/orig{:03}.bmp".format(i*B + j))
                imgs.append(img)
                print(save_path + "/" + dirname + "/" + "orig/orig{:03}.bmp".format(i*B + j))
            # maes = mask_img(imgs, masks)
            # print(maes)
            # if not os.path.exists(save_path + "/" + dirname + "/" + "/meas/"):
            #     os.makedirs(save_path + "/" + dirname + "/" + "meas/")
            # save_img(maes, save_path + "/" + dirname + "/" + "meas/meas{:02}.bmp".format(i))
            count += 1
        print()
    print(count)

recursive_file_check(ROOT_PATH, OUT_PATH)
