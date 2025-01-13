#!/usr/bin/python3
'''
@Version: 0.0
@Autor: wangding
@Date: 2021-01-19-10:36
@Software:PyCharm
LastEditors: wangding wangding19@mails.ucas.ac.cn
LastEditTime: 2025-01-10 16:44:37
@Description: 
    图预处理
'''

import os
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm, trange

from config.Constants import *


# 基于共现构建社交关系，主要针对meme这种用户转发量高而且没有社交图的场景
def ConstructSocialGraph(data_name):
    options=Options(data_name)

    import os
    import pickle

    import numpy as np
    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    user_lists=[[0 for _ in range(len(_idx2ui))] for _ in range(len(_idx2ui))]
    with open(options.train_data) as cas_file:
        with open(options.train_data_id) as id_file:
            id_lines=[line.split("\n")[0] for line in id_file.readlines()]
            cas_lines=[line.split("\n")[0] for line in cas_file.readlines()]
            for i in range(len(cas_lines)):
                line=cas_lines[i]
                id=id_lines[i]
                users=[item.split(",")[0] for item in line.split()]
                for user in users:
                    user_lists[_ui2idx[user]][_ui2idx[id]] = 1
                    user_lists[_ui2idx[id]][_ui2idx[user]] = 1


    print("start counting")
    a=torch.tensor(user_lists,dtype=torch.float32).to(DEVICE)
    res= torch.matmul(a,a)
    res_list=res.detach().cpu().numpy().tolist()
    print("finish counting")

    total=0
    if data_name=="./data/memetracker":
        if os.path.exists(options.net_data): os.remove(options.net_data)
        with open(options.net_data,"a") as file:
            for i in range(2,len(_idx2u)-1):
                for j in range(i+1,len(_idx2u)):
                    if res_list[i][j]>10:
                        if int(_idx2u[i]) < 300000 :
                            total+=2
                            file.write(f"{_idx2u[i]},{_idx2u[j]}\n")
                            file.write(f"{_idx2u[j]},{_idx2u[i]}\n")

    print(total/2)

# 基于共现构建社交关系，主要针对meme这种用户转发量高而且没有社交图的场景 (基于时序排序数据)
def ConstructTrainSocialGraph(data_name,Train_data):
    options=Options(data_name)

    import os
    import pickle

    import numpy as np
    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)
    
    
    user_lists=[[0 for _ in range(len(_idx2ui))] for _ in range(len(_idx2ui))]
    train_users = Train_data[0]
    train_id = Train_data[2]
    for cas, id in zip(train_users,train_id):
        for user in cas:
            if user != 1:
                user_lists[user][id]= 1
                user_lists[id][user]= 1

    print("start counting")
    a=torch.tensor(user_lists,dtype=torch.float32).to(DEVICE)
    res= torch.matmul(a,a)
    res_list=res.detach().cpu().numpy().tolist()
    print("finish counting")

    total=0
    if data_name=="./data/memetracker":
        if os.path.exists(options.net_data): os.remove(options.net_data)
        with open(options.net_data,"a") as file:
            for i in range(2,len(_idx2u)-1):
                for j in range(i+1,len(_idx2u)):
                    if res_list[i][j]>10:
                        if int(_idx2u[i]) < 300000 :
                            total+=2
                            file.write(f"{_idx2u[i]},{_idx2u[j]}\n")
                            file.write(f"{_idx2u[j]},{_idx2u[i]}\n")

    print(f"created social graph with {total/2} edges in total.")

def RefineSocialNetwork(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    social_user_dict = {}
    with open(options.net_data,"r") as file:
        for line in file.readlines():
            user_1, user_2 = line.split("\n")[0].split(",")
            if user_1 not in social_user_dict.keys():
                social_user_dict[user_1] = []

            social_user_dict[user_1].append(user_2)
            # print(social_user_dict[user_1])
    # print(len(social_user_pair))
    # print(social_user_pair)

    cas_user_dict={}
    with open(options.train_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    with open(options.valid_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    with open(options.test_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    output_user_dict = {}
    for user in cas_user_dict.keys(): 
        output_user_dict[user]=[]
        for u in cas_user_dict[user]:
            if user in social_user_dict.keys():
                if u in social_user_dict[user] and u not in output_user_dict[user]:
                    output_user_dict[user].append(u)

    if os.path.exists(options.net_data_refined):os.remove(options.net_data_refined)
    with open(options.net_data_refined,"a") as file:
          for user_1 in output_user_dict.keys():
                for user_2 in output_user_dict[user_1]:
                    file.write(f"{user_1},{user_2}\n") 
                           
def PreprocessDataset(data_name):
    options = Options(data_name)
    
    # 优化社交图
    if os.path.exists(options.net_data_refined) is False and os.path.exists(options.net_data):
        RefineSocialNetwork(data_name)
