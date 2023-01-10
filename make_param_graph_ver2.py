import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def logit(p):
    return np.log(p/(1-p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def show_sigma(p):
    return 255*sigmoid(p)

def show_param(p):
    return relu(p)

projmeth = 'admm'
dataname = 'ex_kobe_method1_lr0005'
parameter_name = 'sigma'

dir_path = "/home/jovyan/workdir/results/" + "trainning_data/" + projmeth + '/'
filename = parameter_name + '_' + dataname
file_path = dir_path + "data_files/" + filename + ".csv"
gragh_path = dir_path + dataname + "/param/"
data = pd.read_csv(file_path, usecols=[1], header=None)
data = pd.Series(data[1])
data = data.str.replace('\[','')
data = data.str.replace('\]','')
data = data.str.split(', ', expand=True)
data = data.astype(float)
# print(data.iloc[-1])
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)
    

fig = plt.figure(0)
if parameter_name == "sigma":
    plt.plot(show_sigma(data.iloc[-1]), linestyle = "-")
else:
    plt.plot(show_param(data.iloc[-1]), linestyle = "-")


plt.title(dataname)
# plt.ylim([0, 0.01])
plt.xlabel("iteration number")
plt.ylabel("parameter")
curr_gragh_path = gragh_path + "{}.png".format(filename)
plt.savefig(curr_gragh_path)

# 1つ用 RSNR
# for index, file_name in enumerate(file_names):
#     fig = plt.figure(index)
#     plt.plot(data[index], linestyle = "-")
#     plt.title(file_name)
#     plt.xlabel("iteration number")
#     plt.ylabel("psnr")
#     curr_gragh_path = gragh_path + "{}.png".format(file_name)
#     plt.savefig(curr_gragh_path)

# 複数用 PSNR
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = file_names[0])
# plt.plot(data[1], linestyle = "--", label = file_names[1])
# plt.plot(data[2], linestyle = ":", label = file_names[2])
# plt.title("method2")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("method2_all")
# plt.savefig(curr_gragh_path)

# FastDVDNet
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.title("FastDVDnet:{}".format(dataname[:-2]))
# plt.ylim([0,35])
# plt.xlim([0,80])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet3")
# plt.savefig(curr_gragh_path)

# FFDnet
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.title("FFDnet:{}".format(dataname[:-2]))
# plt.ylim([0,35])
# plt.xlim([0,80])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("ffdnet3")
# plt.savefig(curr_gragh_path)

# TVで初期化したやつ
# fig = plt.figure(0)
# plt.plot(data[0][:59], linestyle = "-", label = "method1")
# plt.plot(data[1][:59], linestyle = "--", label = "method2")
# plt.plot(data[2][:59], linestyle = ":", label = "method3")
# plt.title("FastDVDnet_tv_initialized:{}".format(dataname[:-2]))
# plt.ylim([0,35])
# plt.xlim([0,60])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet3_tv_initialize1")
# plt.savefig(curr_gragh_path)

# FFDNet TV
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.title("FFDnet_tv_initialized:{}".format(dataname[:-2]))
# plt.ylim([0,40])
# plt.xlim([0,80])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("ffdnet3_tv_initialize5")
# plt.savefig(curr_gragh_path)

# 4つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# # plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.title("FastDVDnet:{}".format(dataname[:-2]))
# plt.ylim([0,35])
# plt.xlim([0,80])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet4")
# plt.savefig(curr_gragh_path)

# FFDnet 4つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# # plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.title("FFDnet:{}".format(dataname[:-2]))
# plt.ylim([0,35])
# plt.xlim([0,80])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("ffdnet4")
# plt.savefig(curr_gragh_path)

# 6つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.plot(data[5], linestyle = (10, (5, 3, 1, 3, 1, 3)), label = "method6")
# plt.title("FastDVDnet:")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_all6")
# plt.savefig(curr_gragh_path)

# 7つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.plot(data[5], linestyle = (10, (5, 3, 1, 3, 1, 3)), label = "method6")
# plt.plot(data[6], linestyle = (15, (5, 3, 1, 3, 1, 3)), label = "method7")
# plt.title("FastDVDnet:")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_all7")
# plt.savefig(curr_gragh_path)

# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method8")
# plt.plot(data[4], linestyle = (10, (5, 3, 1, 3, 1, 3)), label = "method6")
# plt.plot(data[5], linestyle = (15, (5, 3, 1, 3, 1, 3)), label = "method7")
# plt.title("FastDVDnet:")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_all8")
# plt.savefig(curr_gragh_path)

# FFDnet
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method8")
# plt.plot(data[4], linestyle = (10, (5, 3, 1, 3, 1, 3)), label = "method6")
# plt.plot(data[5], linestyle = (15, (5, 3, 1, 3, 1, 3)), label = "method7")
# plt.title("FFDnet")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("ffdnet_all8")
# plt.savefig(curr_gragh_path)

# FFDnet 7つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.plot(data[5], linestyle = (10, (5, 3, 1, 3, 1, 3)), label = "method6")
# plt.plot(data[6], linestyle = (15, (5, 3, 1, 3, 1, 3)), label = "method7")
# plt.title("FFDnet")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("ffdnet_all7")
# plt.savefig(curr_gragh_path)

# sigma可視化用
fig = plt.figure(0)
# sig = [42/(1 + np.exp((i-30)/10)) + 12 for i in range(80)]
sig = [(100*0.5**(i/20)) for i in range(80)]
plt.plot(sig, linestyle = "-")
plt.title("Half life")
plt.ylim([0,110])
plt.xlim([0,80])
plt.xlabel("iteration number")
plt.ylabel("sigma")
curr_gragh_path = "{}.png".format("Half life")
plt.savefig(curr_gragh_path)