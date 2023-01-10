import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

method_name ="ex_kobe_method1_lr0005"
projmeth = 'admm'
accelerate = False
dataname = 'kobe32'

dir_path = "/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/'

if accelerate:
    comp = 'method1_acc'
else:
    comp = 'method1'
file_names = [method_name, comp]
file_paths = [dir_path + filename + '/' + dataname + '/' + 'ssim_' + filename + ".csv" for filename in file_names]
gragh_path = dir_path + method_name + '/' + dataname + '/' + 'graph' + '/' + "ssim/"
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)
data = []

for file_path in file_paths:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data.append(data_n.mean())
  
fig = plt.figure(0)
# 棒の配置位置、ラベルを用意
x = np.array([i for i in range(len(data))])
# x_labels = ["{}".format(i+1) for i in range(len(data))]

x_labels = ['ours', file_names[1]]
 
# マージンを設定
margin = 0.2  #0 <margin< 1
totoal_width = 1 - margin

# 棒グラフをプロット
for i, d in enumerate(data):
  # pos = x - totoal_width *( 1- (2*i+1)/len(data) )/2
  plt.bar(i, d, width=0.7)
  plt.text(i, d, "{:.2f}dB".format(d, ".3f"), ha='center', va='bottom')
 
# ラベルの設定
plt.xticks(x, x_labels)

plt.title(dataname[:-2])
# plt.ylim([0,1])
# plt.legend(loc = "lower right")
plt.xlabel("frame")
plt.ylabel("ssim")
curr_gragh_path = gragh_path + "{}.png".format("ssim_avg")
plt.savefig(curr_gragh_path)

