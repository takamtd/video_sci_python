import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

method_name ="ex_traffic_method1_ver2"
projmeth = 'gap'
dataname = 'traffic48'

dir_path = "/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/'
dir_path += method_name + '/' + dataname + '/' 'test' +'/'
file_names = [method_name, 'method1']
file_paths = [dir_path + 'psnr_' + filename + ".csv" for filename in file_names]
gragh_path = dir_path + "psnr/"
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)
data = []

for file_path in file_paths:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data.append(data_n)
# print(data)
# 複数用 PSNR
fig = plt.figure(0)

# 棒の配置位置、ラベルを用意
x = np.array([i for i in range(len(data[0]))])
x_labels = ["{}".format(i+1) for i in range(len(data[0]))]

labels = ['ours', datanames[1]]
 
# マージンを設定
margin = 0.2  #0 <margin< 1
totoal_width = 1 - margin

# 棒グラフをプロット
for i, d in enumerate(data):
  pos = x - totoal_width *( 1- (2*i+1)/len(data) )/2
  plt.bar(pos, d, width = totoal_width/len(data), label=labels[i])
 
# ラベルの設定
plt.xticks(x, x_labels, rotation=90, fontsize=8)

plt.title("traffic")
# plt.ylim([0,35])
plt.legend(loc = "lower right")
plt.xlabel("frame")
plt.ylabel("psnr")
curr_gragh_path = gragh_path + "{}.png".format("psnr")
plt.savefig(curr_gragh_path)

