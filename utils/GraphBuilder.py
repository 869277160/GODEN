#!/usr/bin/python3
'''
@Version: 0.0
@Autor: wangding
@Date: 2021-01-19-10:36
@Software:PyCharm
LastEditors: wangding wangding19@mails.ucas.ac.cn
LastEditTime: 2025-01-13 16:20:47
@Description: 
    Build means we construct the graph based on certain data and save them in a certain file 
    Create means we create the graph based on cretain data  and output certain data type for our model 
    Load means we read the graph data based on a certain file and output Data class for model 
'''

import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


import config.Constants as Constants
from config.Constants import *
from utils.DataConstructor import Split_data
from utils.GraphPreprocess import PreprocessDataset, RefineSocialNetwork

################## 基于随机数据构建静态超图 ##################

# 基于训练文件构建用户转发图
def BuildRepostGraph(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    train_data = open(options.train_data, "r")
    lines = train_data.readlines()

    if os.path.exists(options.repost_net_data): os.remove(options.repost_net_data)
    with open(options.repost_net_data, "a") as file:
        for i in range(0, len(lines)):
            items = lines[i].split()
            for i in range(0, len(items) - 2):
                user1, _ = items[i].split(",")
                user2, _ = items[i + 1].split(",")
                file.write(f"{user1},{user2}\n")

    train_data.close()
    
# 基于训练文件构建显式超图
def BuildItemGraph(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    train_data = open(options.train_data, "r")
    lines = train_data.readlines()
    train_data_id = open(options.train_data_id, "r")
    ids = [line.split("\n")[0] for line in train_data_id.readlines()]

    if os.path.exists(options.item_net_data): os.remove(options.item_net_data)
    with open(options.item_net_data, "a") as file:
        for i in range(0, len(lines)):
            items = lines[i].split()
            for item in items:
                if item !=  "\n":
                    user, _ = item.split(",")
                    file.write(f"{user},{ids[i]}\n")

    train_data.close()
    train_data_id.close()

################## 基于时序数据构建静态图 ##################

# 基于时序重排数据构建用户转发图
def BuildTrainRepostGraph(data_name,train_data):  
    
    cas = train_data[0]
    idx = train_data[2]
    
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)
    
    if os.path.exists(options.repost_net_data):
        os.remove(options.repost_net_data)
    
    with open(options.repost_net_data, "a") as file:
        for i in range(0, len(cas)):
            for j in range(0, len(cas[i]) - 2):
                # first_user = cas[i][0]
                user1 = cas[i][j]
                user2 = cas[i][j + 1]
                # if user is not EOS:
                # user, _ = item.split(",")
                if (user1 is not EOS and user1 is not PAD) and (user2 is not EOS and user2 is not PAD):
                    file.write(f"{_idx2u[user1]},{_idx2u[user2]}\n")
                    # file.write(f"{_idx2u[user1]},{_idx2u[first_user]}\n")

# 基于时序重排数据构建显式超图
def BuildTrainItemGraph(data_name,train_data):
    
    cas = train_data[0]
    idx = train_data[2]
    
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    if os.path.exists(options.item_net_data):os.remove(options.item_net_data)
    
    with open(options.item_net_data, "a") as file:
        for i in range(0, len(cas)):
            for user in cas[i]:
                if user is not EOS and user is not PAD :
                    # user, _ = item.split(",")
                    file.write(f"{_idx2ui[user]},{_idx2ui[idx[i]]}\n")

################## 整合输出所有异质图 ##################

# 异质图整合
def LoadHeteStaticGraph(data_name, Type, PreProcess=False):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    edges_list = []
    edges_type_list = []
    edges_weight_list = []
    
    if PreProcess:
        PreprocessDataset(data_name=data_name)
        
        if Type.find("item") != -1:
            with open(options.item_net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]
                relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                edge[0] in _ui2idx and edge[1] in _ui2idx]
                # print(relation_list)
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                temp_edges_type_list = [0] * len(relation_list_reverse)
                # print(relation_list_reverse)
                edges_list += relation_list_reverse
                edges_type_list += temp_edges_type_list
                edges_weight_list+=[1.0] * len(relation_list_reverse)
               
        # memetracker 数据集暂时不使用社交网络 
        # if Type.find("social") != -1 and data_name.find("memetracker") == -1:
        if Type.find("social") != -1:
            if os.path.exists(options.net_data) is False or os.path.exists(options.net_data_refined) is False: 
                print("There exists no social grpah!!")
            else:
                if os.path.exists(options.net_data_refined) is False: 
                    RefineSocialNetwork(data_name)
                with open(options.net_data_refined, 'r') as handle:
                    relation_list = handle.read().strip().split("\n")
                    relation_list = [edge.split(',') for edge in relation_list]
                    relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                    edge[0] in _ui2idx and edge[1] in _ui2idx]
                    # print(relation_list)
                    relation_list_reverse = [edge[::-1] for edge in relation_list]
                    temp_edges_type_list = [1] * len(relation_list_reverse)
                    # print(relation_list_reverse)
                    edges_list += relation_list_reverse
                    edges_type_list += temp_edges_type_list
                    edges_weight_list+=[1.0] * len(relation_list_reverse) # social weight 低一些可能更好
            print("load graph!")

        if Type.find("diffusion") != -1:
            with open(options.repost_net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]
                relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                edge[0] in _ui2idx and edge[1] in _ui2idx]
                # print(relation_list)
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                temp_edges_type_list = [2] * len(relation_list_reverse)
                # print(relation_list_reverse)
                edges_list += relation_list_reverse
                edges_type_list += temp_edges_type_list
                edges_weight_list+=[1.0] * len(relation_list_reverse)
    
    else :
        if Type.find("item") != -1:
            with open(options.item_net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]
                relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                edge[0] in _ui2idx and edge[1] in _ui2idx]
                # print(relation_list)
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                temp_edges_type_list = [0] * len(relation_list_reverse)
                # print(relation_list_reverse)
                edges_list += relation_list_reverse
                edges_type_list += temp_edges_type_list
                edges_weight_list+=[1.0] * len(relation_list_reverse)
        
        # memetracker 数据集暂时不使用社交网络 
        if Type.find("social") != -1:
            if os.path.exists(options.net_data) is False: 
                print("There exists no social grpah!!")
            else:
                with open(options.net_data, 'r') as handle:
                    relation_list = handle.read().strip().split("\n")
                    relation_list = [edge.split(',') for edge in relation_list]
                    relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                    edge[0] in _ui2idx and edge[1] in _ui2idx]
                    # print(relation_list)
                    relation_list_reverse = [edge[::-1] for edge in relation_list]
                    temp_edges_type_list = [1] * len(relation_list_reverse)
                    # print(relation_list_reverse)
                    edges_list += relation_list_reverse
                    edges_type_list += temp_edges_type_list
                    edges_weight_list+=[1.0] * len(relation_list_reverse)
            print("load graph!")

        if Type.find("diffusion") != -1:
            with open(options.repost_net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]
                relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                edge[0] in _ui2idx and edge[1] in _ui2idx]
                # print(relation_list)
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                temp_edges_type_list = [2] * len(relation_list_reverse)
                # print(relation_list_reverse)
                edges_list += relation_list_reverse
                edges_type_list += temp_edges_type_list
                edges_weight_list+=[1.0] * len(relation_list_reverse)
    
    edges_list_tensor = torch.LongTensor(edges_list).t().to(DEVICE)
    edges_type = torch.LongTensor(edges_type_list).to(DEVICE)
    edges_weights = torch.FloatTensor(edges_weight_list).to(DEVICE)

    graph = Data(edge_index=edges_list_tensor, edge_type=edges_type, edges_weights=edges_weights)

    return graph



if __name__ == "__main__":
    
    pass 