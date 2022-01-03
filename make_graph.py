import pandas as pd
import matplotlib.pyplot as plt
import os

err_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data//err/err_log0.csv"
sim_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/sim/sim_log0.csv"
kin_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/kin/kin_log0.csv"
sim_angle_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/sim_angle/sim_angle_log0.csv"
gragh_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/gragh/angle/"
os.makedirs(gragh_path)
err_data = pd.read_csv(err_log_path)
sim_data = pd.read_csv(sim_log_path)
kin_data = pd.read_csv(kin_log_path)
sim_angle_data = pd.read_csv(sim_angle_log_path)
# for index, column_name in enumerate(err_data):
#     if index != 0:
#         # print(column_name)
#         fig = plt.figure(index)
#         plt.plot(err_data[column_name], linestyle = "-")
#         plt.title(column_name)
#         plt.xlabel("timestep")
#         plt.ylabel("loss")
#         curr_gragh_path = gragh_path + "{}.png".format(column_name)
#         plt.savefig(curr_gragh_path)

for index, column_name in enumerate(sim_data):
    if index != 0:
        # print(column_name)
        fig = plt.figure(index)
        plt.plot(sim_data[column_name], linestyle = "-", label = "atlas")
        plt.plot(kin_data[column_name], linestyle = "--", label = "mocap")
        # plt.plot(kin_data[column_name]*1.5, linestyle = "--", label = "mocap*1.5")
        plt.legend(loc = "upper right")
        plt.title(column_name)
        plt.xlabel("timestep")
        plt.ylabel("angle")
        curr_gragh_path = gragh_path + "{}.png".format(column_name)
        plt.savefig(curr_gragh_path)
