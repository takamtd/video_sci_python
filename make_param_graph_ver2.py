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
dataname = 'ex_davis_method1'
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
    # plt.plot(show_sigma(data.iloc[403]), linestyle = "-")
else:
    plt.plot(show_param(data.iloc[-1]), linestyle = "-")
    # plt.plot(show_param(data.iloc[135]), linestyle = "-")


plt.title(dataname)
# plt.ylim([0, 0.01])
plt.xlabel("iteration number")
plt.ylabel("parameter")
curr_gragh_path = gragh_path + "{}.png".format(filename)
# curr_gragh_path = gragh_path + "{}.png".format(filename + "135")
# curr_gragh_path = gragh_path + "{}.png".format(filename + "403")
# curr_gragh_path = gragh_path + "{}.png".format(filename + "671")
plt.savefig(curr_gragh_path)

