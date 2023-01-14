import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics

method_name ="ex_davis_method1"
projmeth = 'admm'
# datanames = ['kobe32','runner40','drop40','crash32','aerial32']
test_datanames = ['kobe32','traffic48', 'runner40','drop40','crash32','aerial32']
accelerate = False

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
dirnames = dirnames[-9:]
for test_dataname in test_datanames:
    dirnames.append(test_dataname)
dirnames.sort()
if accelerate:
    comp = 'method1_acc'
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

# 複数用 PSNR
fig = plt.figure(0)
plt.plot(data1, linestyle = "-", label = 'ours')
plt.plot(data2, linestyle = "--", label = 'Yuan[1]')
# plt.ylim([0,35])
plt.legend(loc = "lower right")
plt.xlabel("iteration number")
plt.ylabel("psnr")
curr_graph_path = graph_path + "{}.png".format("psnrall")
plt.savefig(curr_graph_path)
