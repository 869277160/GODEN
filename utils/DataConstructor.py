import random
import numpy as np
import torch
from torch.autograd import Variable
# import Constants
import pickle

import config.Constants as Constants
from config.Constants import Options, DEVICE

# split the data and process them into indexs for the model input
def Split_data(data_name, train_rate =0.8, valid_rate = 0.1, random_seed = Constants.SEED, load_dict=True, with_EOS=True, max_len = 500):
        options = Options(data_name)
        u2idx = {}
        idx2u = []
        ui2idx = {}
        idx2ui = []
        
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
        user_size = len(u2idx)
        with open(options.ui2idx_dict, 'rb') as handle:
            ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            idx2ui = pickle.load(handle)
        # user_size = len(u2idx)
        
        user_count = [1] * user_size
        t_cascades = []
        timestamps = []
        for line in open(options.data):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    # Twitter,Douban
                    if len(chunk.split(",")) ==2:
                        user, timestamp = chunk.split(",")
                    # Android,Christianity
                    # elif len(chunk.split())==3:
                    #     root, user, timestamp = chunk.split()                                           
                    #     if root in u2idx:          
                    #         userlist.append(u2idx[root])                        
                    #         timestamplist.append(float(timestamp))
                except:
                    print(chunk)
                if user in u2idx:
                    userlist.append(u2idx[user])
                    user_count[u2idx[user]] += 1
                    timestamplist.append(float(timestamp))
            
            # if len(userlist) > max_len:
            if len(userlist) > max_len and len(userlist) <= 500:    
                userlist = userlist[:max_len]
                timestamplist = timestamplist[:max_len]
                
            # if len(userlist) <= max_len:
            if len(userlist) >= 2 and len(userlist) <= max_len:
                if with_EOS:
                    userlist.append(Constants.EOS)
                    timestamplist.append(Constants.EOS)
                t_cascades.append(userlist)
                timestamps.append(timestamplist)
                
        # read all ids 
        t_cascades_ids = []
        for line in open(options.data_id):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip()        
            t_cascades_ids.append(ui2idx[(chunks)]) 
            
            
        
        '''ordered by timestamps'''        
        order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
        timestamps = sorted(timestamps)
        t_cascades[:] = [t_cascades[i] for i in order]
        cas_idx =  [t_cascades_ids[i] for i in order]
        
        '''data split'''
        train_idx_ = int(train_rate*len(t_cascades))
        train = t_cascades[0:train_idx_]
        train_t = timestamps[0:train_idx_]
        train_idx = cas_idx[0:train_idx_]
        train = [train, train_t, train_idx]
        
        valid_idx_ = int((train_rate+valid_rate)*len(t_cascades))
        valid = t_cascades[train_idx_:valid_idx_]
        valid_t = timestamps[train_idx_:valid_idx_]
        valid_idx = cas_idx[train_idx_:valid_idx_]
        valid = [valid, valid_t, valid_idx]
        
        test = t_cascades[valid_idx_:]
        test_t = timestamps[valid_idx_:]
        test_idx = cas_idx[valid_idx_:]
        test = [test, test_t, test_idx]
            
        total_len =  sum(len(i)-1 for i in t_cascades)
        train_size = len(train_t)
        valid_size = len(valid_t)
        test_size = len(test_t)
        print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
        print("total size:%d " %(len(t_cascades)))
        print("average length:%f" % (total_len/len(t_cascades)))
        print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
        print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))    
        print("user size:%d"%(user_size-2))           
        
        return user_size, t_cascades, timestamps, train, valid, test,
    


# process the data into batches and pad them
class DataConstructor(object):
    ''' For data iteration ''' 

    def __init__(self, cas, batch_size=64, load_dict=True, cuda=True,  test=False, with_EOS=True,_need_shsuffle=False, max_len =500): 
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.test = test
        self.with_EOS = with_EOS          
        self.cuda = cuda
        self._need_shuffle = _need_shsuffle
        self.max_len = max_len 
        
        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

        #  regroup the training data 
        if self._need_shuffle:
            num = [x for x in range(0, len(self.cas))]
            random_seed_int = random.randint(0, 1000)
            print(f"Init Dataset and shuffle data with {random_seed_int}")
            random.seed(random_seed_int)
            random.shuffle(num)
            self.cas = [self.cas[num[i]] for i in range(0, len(num))]
            self.time = [self.time[num[i]] for i in range(0, len(num))]
            self.idx = [self.idx[num[i]] for i in range(0, len(num))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''
            
            # NOTE: this is the orginal setting for the model
            ''' Pad  the instance in batch to the max seq length in batch '''
            # max_len = max(len(inst) for inst in insts)
            max_len = 205
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
                
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.to(DEVICE)

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            if self.cuda:
                seq_idx = Variable(torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test).to(DEVICE)  
            else :
                seq_idx = Variable(torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)
            return seq_data, seq_data_timestamp, seq_idx
        else:
            if self._need_shuffle:
                num = [x for x in range(0, len(self.cas))]
                random_seed_int = random.randint(0, 1000)
                print(f"shuffle data with {random_seed_int}")
                random.seed(random_seed_int)
                random.shuffle(num)
                self.cas = [self.cas[num[i]] for i in range(0, len(num))]
                self.time = [self.time[num[i]] for i in range(0, len(num))]
                self.idx = [self.idx[num[i]] for i in range(0, len(num))]


            self._iter_count = 0
            raise StopIteration()
