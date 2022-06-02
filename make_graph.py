import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = "/home/jovyan/workdir/results/savedmat"
# alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
dataname = 'aerial32'
data_path = "/grayscale/"+dataname+"/data/"
file_names = ["gapffdnet_psnr_method1_60times", "gapffdnet_psnr_method2_60times", "gapffdnet_psnr_method3_60times"]
# file_names = ["gapfastdvdnet_psnr_method1", "gapfastdvdnet_psnr_method2", "gapfastdvdnet_psnr_method3"]
# file_names = ["gapfastdvdnet_psnr_method1_tv_initialize1", "gapfastdvdnet_psnr_method2_tv_initialize1", "gapfastdvdnet_psnr_method3_tv_initialize1"]
# file_names = ["gapfastdvdnet_psnr_method1_tv_initialize5", "gapfastdvdnet_psnr_method2_tv_initialize5", "gapfastdvdnet_psnr_method3_tv_initialize5"]

# file_names = ["gapfastdvdnet_psnr_method1", "gapfastdvdnet_psnr_method2", "gapfastdvdnet_psnr_method3", "gapfastdvdnet_psnr_method4", "gapfastdvdnet_psnr_method5"]
# file_names = ["gapfastdvdnet_psnr_method1_tv_initialize1", "gapfastdvdnet_psnr_method2_tv_initialize1", "gapfastdvdnet_psnr_method3_tv_initialize1", "gapfastdvdnet_psnr_method4_tv_initialize1", "gapfastdvdnet_psnr_method5_tv_initialize1"]

# file_names = ["gapfastdvdnet_psnr_method1", "gapfastdvdnet_psnr_method2", "gapfastdvdnet_psnr_method3", "gapfastdvdnet_psnr_method4", "gapfastdvdnet_psnr_method5", "gapfastdvdnet_psnr_method6", "gapfastdvdnet_psnr_method7"]
# file_names = ["gapfastdvdnet_psnr_method1", "gapfastdvdnet_psnr_method2", "gapfastdvdnet_psnr_method3", "gapfastdvdnet_psnr_method8", "gapfastdvdnet_psnr_method6", "gapfastdvdnet_psnr_method7"]

# file_names = ["gapfastdvdnet_psnr_method2", "gapffdnet_psnr_method2", "gaptv_psnr_method2"]
# file_names = ["gapfastdvdnet_psnr_method2_tv_initialize1", "gapffdnet_psnr_method2_tv_initialize1", "gaptv_psnr_method2_tv_initialize1"]
file_paths = [dir_path + data_path + name + ".csv" for name in file_names]
gragh_path = dir_path + data_path + "/gragh/psnr/"
if not os.path.exists(gragh_path):
    os.makedirs(gragh_path)
data = []

for file_path in file_paths:
    data.append(pd.read_csv(file_path))

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
# plt.xlim([0,60])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet3_60times")
# plt.savefig(curr_gragh_path)

# FFDnet
fig = plt.figure(0)
plt.plot(data[0], linestyle = "-", label = "method1")
plt.plot(data[1], linestyle = "--", label = "method2")
plt.plot(data[2], linestyle = ":", label = "method3")
plt.title("FFDnet:{}".format(dataname[:-2]))
plt.ylim([0,35])
plt.xlim([0,60])
plt.legend(loc = "lower right")
plt.xlabel("iteration number")
plt.ylabel("psnr")
curr_gragh_path = gragh_path + "{}.png".format("ffdnet3_60times")
plt.savefig(curr_gragh_path)

# TVで初期化したやつ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.title("FastDVDnet:tv_initialize1")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_tv_initialize1_all")
# plt.savefig(curr_gragh_path)

# 4つ
# fig = plt.figure(0)
# plt.plot(data[0], linestyle = "-", label = "method1")
# plt.plot(data[1], linestyle = "--", label = "method2")
# plt.plot(data[2], linestyle = ":", label = "method3")
# plt.plot(data[3], linestyle = "-.", label = "method4")
# plt.plot(data[4], linestyle = (0, (5, 3, 1, 3, 1, 3)), label = "method5")
# plt.title("FastDVDnet:")
# plt.ylim([0,35])
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_all5")
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
# fig = plt.figure(0)
# sig = [42/(1 + np.exp((i-30)/10)) + 12 for i in range(80)]
# # sig    = [50*0.97**i for i in range(80)]
# plt.plot(sig, linestyle = "-")
# plt.title("sigmoid")
# plt.ylim([0,55])
# plt.xlabel("iteration number")
# plt.ylabel("sigma")
# curr_gragh_path = "{}.png".format("sigmoid")
# plt.savefig(curr_gragh_path)