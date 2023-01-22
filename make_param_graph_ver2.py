import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
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

projmeth = 'gap'
dataname = 'davis_train_add_noise'
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
    
x = [i+1 for i in range(60)]
fig = plt.figure(0)
if parameter_name == "sigma":
    plt.plot(x, show_sigma(data.iloc[-1]), linestyle = "-")
    # plt.plot(show_sigma(data.iloc[403]), linestyle = "-")
else:
    plt.plot(x, show_param(data.iloc[-1]), linestyle = "-")
    # plt.plot(show_param(data.iloc[135]), linestyle = "-")

plt.rcParams["font.size"] = 18
plt.ylim([0, 150])
plt.xlim([0, 60])
plt.xlabel("反復回数$k$", fontsize=20)
plt.ylabel("パラメータ$\sigma_k$", fontsize=20)
plt.tight_layout()
curr_gragh_path = gragh_path + "{}.png".format(filename)
# curr_gragh_path = gragh_path + "{}.png".format(filename + "135")
# curr_gragh_path = gragh_path + "{}.png".format(filename + "403")
# curr_gragh_path = gragh_path + "{}.png".format(filename + "671")
plt.savefig(curr_gragh_path)

