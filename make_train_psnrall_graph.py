import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
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
file_paths1 = [dir_path + dirname + '/' + 'psnrall_' + method_name + ".csv" for dirname in dirnames[:-9]]
file_paths2 = ["/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/' + comp + "/" + dirname + '/' + 'psnrall_' + comp + ".csv" for dirname in dirnames[:-9]]

graph_path = dir_path + 'trained' + '/' + "psnrall/"
if not os.path.exists(graph_path):
    os.makedirs(graph_path)
data1 = []
data2 = []

for file_path in file_paths1:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data1.append(data_n)

for file_path in file_paths2:
    data_n = pd.read_csv(file_path, header=None)
    data_n = pd.Series(data_n[0])
    # print(type(data_n[0]))
    data2.append(data_n)

data1 = pd.DataFrame(data1)
data1 = data1.mean(axis=0)
data2 = pd.DataFrame(data2)
data2 = data2.mean(axis=0)
x = [i+1 for i in range(60)]
# 複数用 PSNR
fig = plt.figure(0)
plt.rcParams["font.size"] = 16
plt.plot(x, data1, linestyle = "-", label = '提案手法')
plt.plot(x, data2, linestyle = "--", label = '従来手法')
plt.ylim([0, 35])
plt.xlim([0, 60])
plt.legend(loc = "lower right")
plt.xlabel("反復回数$k$", fontsize=18)
plt.ylabel("PSNR[dB]", fontsize=18)
plt.tight_layout()
curr_graph_path = graph_path + "{}.png".format("psnrall")
plt.savefig(curr_graph_path)
