import pandas as pd
import matplotlib.pyplot as plt
import os

dir_path = "/home/jovyan/workdir/results/savedmat"
data_path = "/grayscale/runner40/data/"
file_names = ["gapfastdvdnet_psnr_method1", "gapfastdvdnet_psnr_method2", "gapfastdvdnet_psnr_method3"]
# file_names = ["gapfastdvdnet_psnr_method3", "gapffdnet_psnr_method3", "gaptv_psnr_method3"]
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
# plt.title("method3")
# plt.legend(loc = "lower right")
# plt.xlabel("iteration number")
# plt.ylabel("psnr")
# curr_gragh_path = gragh_path + "{}.png".format("method3_all")
# plt.savefig(curr_gragh_path)

fig = plt.figure(0)
plt.plot(data[0], linestyle = "-", label = "method1")
plt.plot(data[1], linestyle = "--", label = "method2")
plt.plot(data[2], linestyle = ":", label = "method3")
plt.title("FastDVDnet")
plt.legend(loc = "lower right")
plt.xlabel("iteration number")
plt.ylabel("psnr")
curr_gragh_path = gragh_path + "{}.png".format("fastdvdnet_all")
plt.savefig(curr_gragh_path)