import os, sys, time

import numpy as np
import torch
import torch.nn as nn

import config.Constants as Constants
from config.Constants import (ConfigFromParser, Options, DEVICE)

from models.GODEN import GODEN

from utils.DataConstructor import DataConstructor, Split_data
from utils.GraphBuilder import *
from utils.GraphPreprocess import ConstructTrainSocialGraph, PreprocessDataset
from utils.LossUtil import LossComputing
from utils.Metrics import Metrics
from utils.Optim import TunedScheduledOptim


def train_epoch(model, training_data, optimizer, epoch, criterion, static_graph, diffusion_graph):
    ''' Epoch operation in training phase'''
    
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0
    
    LossFunction = criterion

    for i, batch in enumerate(training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp, tgt_id = batch
        tgt.to(DEVICE); tgt_timestamp.to(DEVICE); tgt_id.to(DEVICE)
        user_gold = tgt[:, 1:].to(DEVICE)

        # start_time = time.time()
        n_words = user_gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        
        optimizer.zero_grad()
            
        user_pred = model(tgt, 
                          tgt_timestamp, 
                          tgt_id,
                          epoch = epoch, 
                          train=True, 
                          static_graph=static_graph, 
                          diffusion_graph = diffusion_graph)
        
        # get loss and backward
        loss, n_correct, same_user, input_users = LossFunction(user_pred, user_gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate(epoch)

        # note keeping
        batch_num += tgt.size(0)
        n_total_correct += n_correct
        total_loss = total_loss + loss.item()
        total_same_user += same_user
        n_total_uniq_user += input_users
        
        print("Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item() / len(user_pred)),
              f"\t\toutput_users:{(same_user)}/{(input_users)}={same_user / input_users}", )

    return total_loss / n_total_words, n_total_correct / n_total_words, total_same_user / n_total_uniq_user

def test_epoch(model, validation_data, epoch, static_graph, diffusion_graph, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    
    metric = Metrics()
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
        
    n_total_words = 0
    for i, batch in enumerate(validation_data):  
        print("Validation batch ", i)
        # prepare data
        tgt, tgt_timestamp, tgt_id = batch
        tgt.to(DEVICE); tgt_timestamp.to(DEVICE); tgt_id.to(DEVICE)
        user_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()
        
        user_pred = model(tgt, 
                          tgt_timestamp, 
                          tgt_id, 
                          epoch = epoch, 
                          train=False, 
                          static_graph=static_graph, 
                          diffusion_graph = diffusion_graph)

        user_pred = user_pred.detach().cpu().numpy()
        scores_batch, scores_len = metric.compute_metric(user_pred, user_gold, k_list)

        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores

def train_model(data_path):

    # ========= Preparing Data with DataConstructer =========#
    # prepare the data that sorted with time 
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_ratio, opt.valid_ratio, load_dict=True, max_len=opt.max_len)
    
    # split the data based on the file
    train_data = DataConstructor(train, batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=True , max_len = opt.max_len)
    valid_data = DataConstructor(valid, batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=False, max_len = opt.max_len)
    test_data  = DataConstructor(test , batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=False, max_len = opt.max_len)
    opt.user_size = user_size
    option = Options(opt.data_path)
    _ui2idx = {}
    with open(option.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    opt.token_size = len(_ui2idx)
    
    # Build the corresponding graphs with generated data.
    BuildTrainRepostGraph(opt.data_path, train)
    BuildTrainItemGraph(opt.data_path, train) 
    
    # Build the social graph for non-social datasets
    # 构建社交图
    if opt.graph_preprocess: 
        options = Options()
        if opt.data == "memetracker" and not os.path.exists(options.net_data):
            ConstructTrainSocialGraph(opt.data_path,train)
        if os.path.exists(options.net_data_refined) is False and os.path.exists(options.net_data):
            RefineSocialNetwork(opt.data_path)
        PreprocessDataset(opt.data_path)
    
    
    # ========= Preparing Graph =========#
    static_graph = LoadHeteStaticGraph(opt.data_path, Type=opt.static_graph_type, PreProcess=opt.graph_preprocess) 
    dynamic_graph = LoadHeteStaticGraph(opt.data_path,Type="item+diffusion", PreProcess=opt.graph_preprocess)  
    
    # ========= Preparing Model =========#
    model = GODEN(opt)
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    params = filter(lambda p: p.requires_grad, model.parameters())
    Origin_optimizer = torch.optim.Adam(params, betas=(0.9, 0.98),weight_decay = opt.l2, eps=1e-09)  # weight_decay is l2 regularization
    optimizer = TunedScheduledOptim(Origin_optimizer, opt.d_model, opt.n_warmup_steps, data_path) 
    criterion = LossComputing(opt)

    if torch.cuda.is_available():
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
    
    validation_history = 0.0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, optimizer,epoch_i,criterion,static_graph,dynamic_graph)
        print('  - (Training)   loss: {loss: 4.3f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60))
        
        if epoch_i >= 0:
            start = time.time()
            val_scores = test_epoch(model, valid_data, epoch_i, static_graph, dynamic_graph)
            
            # validate the models
            print('  - ( Validation )) ')
            for metric in val_scores.keys():
                print(metric + ' ' + str(val_scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            # test the model
            print('  - (Test) ')
            scores = test_epoch(model, test_data, epoch_i, static_graph, dynamic_graph)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            
            # save the best model results 
            if validation_history <= scores["hits@100"]:
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                validation_history = scores["hits@100"]
                best_scores = scores
                print("Save best model!!!") 
                torch.save(model.state_dict(), opt.save_path)

    print(" -(Finished!!) \n Best scores: ")        
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))

def test_model(data_path):
    
    # ========= Preparing Data with DataConstructor =========#
    # prepare the data that sorted with time 
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_ratio, opt.valid_ratio, load_dict=True)
    
    # split the data based on the file
    train_data = DataConstructor(train, batch_size=opt.batch_size, load_dict=True, cuda=False, max_len = opt.max_len)
    valid_data = DataConstructor(valid, batch_size=opt.batch_size, load_dict=True, cuda=False, max_len = opt.max_len)
    test_data  = DataConstructor(test , batch_size=opt.batch_size, load_dict=True, cuda=False, max_len = opt.max_len)  
    
    # Build the corresponding graphs with generated data.
    BuildTrainRepostGraph(opt.data_path, train)
    BuildTrainItemGraph(opt.data_path, train)   

    # Build the social graph for non-social datasets
    # 为没有社交图的数据集利用共现关系构建社交图
    if opt.graph_preprocess: 
        options = Options()
        if opt.data == "memetracker" and not os.path.exists(options.net_data):
            ConstructTrainSocialGraph(opt.data_path,train)
        if os.path.exists(options.net_data_refined) is False and os.path.exists(options.net_data):
            RefineSocialNetwork(opt.data_path)
        PreprocessDataset(opt.data_path)
    
    # ========= Preparing Graphs =========#
    static_graph = LoadHeteStaticGraph(opt.data_path, Type=opt.static_graph_type, PreProcess=opt.graph_preprocess) 
    diffusion_graph = LoadHeteStaticGraph(opt.data_path, Type=opt.dynamic_graph_type, PreProcess=opt.graph_preprocess)  

    # ========= Preparing Models =========#
    model = GODEN(opt)
    model.load_state_dict(torch.load(opt.save_path))
        
    if torch.cuda.is_available():
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
    
    scores = test_epoch(model, test_data, opt.epoch, static_graph, diffusion_graph)
    print('  - (Test) ')
    for metric in scores.keys():
        
        print(metric + ' ' + str(scores[metric]))

if __name__ == "__main__":
    
    opt = ConfigFromParser()

    # 训练模型
    train_model(opt.data_path)