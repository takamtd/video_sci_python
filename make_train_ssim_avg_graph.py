import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics

projmeth = 'gap'
accelerate = False
noise = 0
# method_name = "ex_davis_method1"
method_name = "davis_train_add_noise"
# method_name = "davis_train_add_noise5"

test_datanames = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
dir_path = "/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/'
dir_path += method_name + '/'

dirnames = os.listdir(dir_path)
if 'trained' in dirnames:
    dirnames.remove('trained')
if 'test' in dirnames:
    dirnames.remove('test')
for test_dataname in test_datanames:
    dirnames.remove(test_dataname)
dirnames.sort()

if accelerate:
    if noise == 1:
        comp = 'method1_acc_add_meas_noise'
    elif noise == 5:
        comp = 'method1_acc_add_meas_noise5'
    else:
        comp = 'method1_acc'
else:
    if noise == 1:
        comp = 'method1_add_meas_noise'
    elif noise == 5:
        comp = 'method1_add_meas_noise5'
    else:
        comp = 'method1'
file_names = [method_name, comp]
file_paths1 = [dir_path + dirname + '/' + 'ssim_' + method_name + ".csv" for dirname in dirnames[:-9]]
file_paths2 = ["/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/' + comp + "/" + dirname + '/' + 'ssim_' + comp + ".csv" for dirname in dirnames[:-9]]

gragh_path = dir_path + 'trained' + '/' + "ssim/"
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)
data1 = []
data2 = []

for file_path in file_paths1:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data1.append(data_n.mean())
for file_path in file_paths2:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data2.append(data_n.mean())

data1 = statistics.mean(data1)
data2 = statistics.mean(data2)
data = [data1, data2]

# print(data)
# 複数用 PSNR
fig = plt.figure(0)

# 棒の配置位置、ラベルを用意
x = np.array([i for i in range(len(data))])
# x_labels = ["{}".format(i+1) for i in range(len(data))]

x_labels = ['ours', 'Yuan[1]']
 
# マージンを設定
margin = 0.2  #0 <margin< 1
totoal_width = 1 - margin

# 棒グラフをプロット
for i, d in enumerate(data):
    # pos = x - totoal_width *( 1- (2*i+1)/len(data) )/2
    plt.bar(i, d, width=0.7)
    plt.text(i, d, "{:.4f}".format(d, ".3f"), ha='center', va='bottom')
 
# ラベルの設定
plt.xticks(x, x_labels)

# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("frame")
plt.ylabel("SSIM")
curr_gragh_path = gragh_path + "{}.png".format("ssim_avg")
plt.savefig(curr_gragh_path)

