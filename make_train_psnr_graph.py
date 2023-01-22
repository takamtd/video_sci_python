import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics

method_name ="ex_davis_method1_acc"
projmeth = 'gap'
# datanames = ['kobe32','runner40','drop40','crash32','aerial32']
test_datanames = ['kobe32','traffic48', 'runner40','drop40','crash32','aerial32']
accelerate = True

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
    comp = 'method1_acc'
else:
    comp = 'method1'
file_names = [method_name, comp]
file_paths1 = [dir_path + dirname + '/' + 'psnr_' + method_name + ".csv" for dirname in dirnames[:-9]]
file_paths2 = ["/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/' + comp + "/" + dirname + '/' + 'psnr_' + comp + ".csv" for dirname in dirnames[:-9]]

gragh_path = dir_path + 'trained' + '/' + "psnr/"
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

data = [data1, data2]
print([(d1-d2)<0 for d1,d2 in zip(data1,data2)])
# 複数用 PSNR
# fig = plt.figure(0)

# # 棒の配置位置、ラベルを用意
# x = np.array([i for i in range(len(data1))])
# x_labels = datanames

# labels = ['ours', 'method1']
 
# # マージンを設定
# margin = 0.2  #0 <margin< 1
# totoal_width = 1 - margin

# # 棒グラフをプロット
# for i, d in enumerate(data):
#   pos = x - totoal_width *( 1- (2*i+1)/len(data) )/2
#   plt.bar(pos, d, width = totoal_width/len(data), label=labels[i])
#   for p, d_t in zip(pos, d):
#     plt.text(p, d_t, "{:.2f}".format(d_t, ".3f"), ha='center', va='bottom', fontsize=9)
 
# # ラベルの設定
# plt.xticks(x, x_labels, rotation=45, fontsize=8)

# plt.title("train")
# # plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("frame")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("psnr")
# plt.savefig(curr_gragh_path)

