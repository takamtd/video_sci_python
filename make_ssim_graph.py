import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

projmeth = 'gap'
dataname = 'ex_traffic_method1_acc'

dir_path = "/home/jovyan/workdir/results/" + "trainning_data/" + projmeth + '/'
filename = 'ssim_' + dataname
file_path = dir_path + "data_files/" + filename + ".csv"
gragh_path = dir_path + dataname + "/ssim/" +"/"
data = pd.read_csv(file_path, usecols=[1], header=None)
data = pd.Series(data[1])
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)

fig = plt.figure(0)
plt.plot(data, linestyle = "-")
plt.title(dataname)
# plt.ylim([0, 1])
plt.xlabel("epoch")
plt.ylabel("ssim")
curr_gragh_path = gragh_path + "{}.png".format(dataname)
plt.savefig(curr_gragh_path)
