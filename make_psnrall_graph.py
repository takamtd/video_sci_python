import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics

projmeth = 'gap'
dataname = 'ex_davis_method1_acc'
accelerate = False

datanames = ['kobe32', 'traffic48', 'runner40','drop40','crash32','aerial32']
# datanames = ['aerial32']
dir_path = "/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/'
dir_path += method_name + '/'
if accelerate:
    comp = 'method1_acc'
else:
    comp = 'method1'
file_names = [method_name, comp]
file_paths1 = [dir_path + dataname + '/' + 'psnrall_' + method_name + ".csv" for dataname in datanames]
file_paths2 = ["/home/jovyan/workdir/results/savedmat/grayscale/" + projmeth + '/' + comp + "/" + dataname + '/' + 'psnrall_' + comp + ".csv" for dataname in datanames]

graph_paths = [dir_path + dataname + '/' + 'graph/' + 'psnrall/' for dataname in datanames]

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


for graph_path in graph_paths:
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

# 複数用 PSNR
for i, dataname in enumerate(datanames):
    fig = plt.figure(i)
    plt.plot(data1[i], linestyle = "-", label = 'ours')
    plt.plot(data2[i], linestyle = "--", label = 'method1')
    plt.title(dataname[:-2])
    # plt.ylim([0,35])
    plt.legend(loc = "lower right")
    plt.xlabel("iteration number")
    plt.ylabel("psnr")
    curr_graph_path = graph_paths[i] + "{}.png".format("psnrall")
    print(curr_graph_path)
    plt.savefig(curr_graph_path)
