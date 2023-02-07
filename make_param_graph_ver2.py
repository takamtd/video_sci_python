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
accelerate = True
if accelerate:
    datanames = [
        'ex_davis_method1_acc', 'davis_acc_train_add_noise', 'davis_acc_train_add_noise5'
    ]
else:
    datanames = [
        'ex_davis_method1', 'davis_train_add_noise', 'davis_train_add_noise5'
    ]
parameter_name = 'sigma'

for n, dataname in enumerate(datanames):
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
    
    plt.rcParams["font.size"] = 16
    x = [i+1 for i in range(60)]
    y = [50,50,25,12]
    x2 = [0,20,40,60]
    fig = plt.figure(n)
    if parameter_name == "sigma":
        plt.plot(x, show_sigma(data.iloc[-1]), linestyle = "-", label = '学習で得られたパラメータ')
        plt.step(x2, y, linestyle = "--", label = '学習初期値')
        # plt.plot(show_sigma(data.iloc[403]), linestyle = "-")
    else:
        plt.plot(x, show_param(data.iloc[-1]), linestyle = "-")
        # plt.plot(show_param(data.iloc[135]), linestyle = "-")

    plt.ylim([0, 170])
    plt.xlim([0, 60])
    plt.xlabel("反復回数$k$", fontsize=18)
    plt.ylabel("パラメータ$\sigma_k$", fontsize=18)
    plt.legend(loc = "upper right", fontsize=14)
    plt.tight_layout()
    plt.grid()
    curr_gragh_path = gragh_path + "{}.png".format(filename)
    # curr_gragh_path = gragh_path + "{}.png".format(filename + "135")
    # curr_gragh_path = gragh_path + "{}.png".format(filename + "403")
    # curr_gragh_path = gragh_path + "{}.png".format(filename + "642")
    plt.savefig(curr_gragh_path)

