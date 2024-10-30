from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


from minigpt4.models.rec_model import MatrixFactorization
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 
# from minigpt4.models.rec_base_models import hashGNN
import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'



from minigpt4.tasks import base_task
import time
import numpy as np
# import ray

# class my_sign(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, inputs):
#         re = torch.ones_like(inputs)
#         re[inputs<0]= -1.0
#         return re
#         # return (ctx>0).float() 
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         return grad_outputs

# class binarize(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
    
#     def forward(self,x):
#         re = my_sign.apply(x)
#         return re 



# class hashGNN(nn.Module):
#     def __init__(self, config, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.rec_model = MatrixFactorization(config=config)
#         self.emb_dim = config.embedding_size
#         self.proj_u = nn.Linear(self.emb_dim,self.emb_dim)
#         self.proj_i = nn.Linear(self.emb_dim,self.emb_dim)
#         self.act = nn.Tanh()
#         self.binarize = binarize()
    
#     def beta_init(self,beta_step=0.02):
#         self.beta = 0.0
#         self.beta_step=beta_step
#         self.beta_updatable = True

#     def update_beta(self):
#         if self.beta_updatable and self.beta < 1.01:
#             self.beta += self.beta_step
#         else:
#             self.beta_updatable = False
#             self.beta = 1.0
    
#     def computer(self):
#         u_emb = self.rec_model.user_embedding.weight
#         i_emb = self.rec_model.item_embedding.weight
#         u_z = self.act(self.proj_u(u_emb))
#         i_z = self.act(self.proj_i(i_emb))
#         self.hash_u = self.binarize(u_z)
#         self.hash_i = self.binarize(i_z)
    
#     def user_encoder(self,idx):
#         return self.hash_u[idx]
#     def item_encoder(self,idx):
#         return self.hash_u[idx]


#     def forward(self,user,item):
#         u_emb = self.rec_model.user_encoder(user)
#         i_emb = self.rec_model.item_encoder(item)

#         z_u = self.act(self.proj_u(u_emb))
#         z_i = self.act(self.proj_i(i_emb))

#         h_u = self.binarize(z_u)
#         h_i = self.binarize(z_i)

#         h_u_shape = h_u.shape

#         # selector = self.beta < torch.rand_like(h_u) 
#         # o_u = torch.where(selector, z_u, h_u)
#         # o_i = torch.where(selector, z_i, h_i)
#         if self.training:
#             # selector = self.beta < torch.rand([np.prod(h_u_shape[:-1])]) 
#             # selector = selector.reshape(-1,1).repeat(1,h_u_shape[-1]).cuda()
#             selector = self.beta < torch.rand_like(h_u) 
#             o_u = torch.where(selector, z_u, h_u)
#             o_i = torch.where(selector, z_i, h_i)
#         else:
#             o_u = h_u
#             o_i = h_i
#             print("o_u:",o_u)
#         matching = torch.mul(o_u,o_i).sum(dim=-1)
#         return matching

class my_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        re = torch.ones_like(inputs)
        re[inputs<0]= -1.0
        return re
        # return (ctx>0).float() 
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class binarize(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x):
        re = my_sign.apply(x)
        return re 



class hashGNN(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rec_model = MatrixFactorization(config=config)
        self.emb_dim = config.embedding_size
        self.proj_u = nn.Linear(self.emb_dim,self.emb_dim)
        self.proj_i = nn.Linear(self.emb_dim,self.emb_dim)
        self.act = nn.Tanh()
        self.binarize = binarize()
    
    def beta_init(self):
        self.beta = 0.0
        self.beta_updatable = True

    def update_beta(self):
        if self.beta_updatable and self.beta < 1.01:
            self.beta += 0.02
        else:
            self.beta_updatable = False
            self.beta = 1.1
        print(self.beta)
    
    def computer(self):
        u_emb = self.rec_model.user_embedding.weight
        i_emb = self.rec_model.item_embedding.weight
        u_z = self.act(self.proj_u(u_emb))
        i_z = self.act(self.proj_i(i_emb))
        self.hash_u = self.binarize(u_z)
        self.hash_i = self.binarize(i_z)
    
    def hash2str(self, mode='binary'):
        hash_u, hash_i = self.hash_u.int(), self.hash_i.int()
        hash_u[hash_u<0] = 0
        hash_i[hash_i<0] = 0
        def arr2b(arr):
            arr = ["".join(map(str,x)) for x in arr.tolist()]
            return arr
        if mode == 'binary':
            hash_u = arr2b(hash_u)
            hash_i = arr2b(hash_i)
        elif mode == 'ipv4':
            def arr2ipv4(arr):
                arr = torch.chunk(arr,4,dim=1)
                arr = [arr2b(arr_) for arr_ in arr]
                num = len(arr)
                arr_n = [".".join([str(int(arr[n][k],2)) for n in range(num)]) for k in range(len(arr[0]))]
                return arr_n
            hash_u = arr2ipv4(hash_u)
            hash_i = arr2ipv4(hash_i)
        else:
            raise ValueError("Unknow type....", mode)
        return hash_u, hash_i     
    
    def unique_hash(self):
        self.computer()
        def to_bi(arr):
            arr = arr.int().tolist()
            return ["".join([str(x_) for x_ in x]) for x in arr]
        print("unique hash user:", len(set(to_bi(self.hash_u))))
        print("unique hash item:", len(set(to_bi(self.hash_i))))

    
    def user_encoder(self,idx):
        return self.hash_u[idx]
    def item_encoder(self,idx):
        return self.hash_u[idx]
    
    def do_rec(self, user, cliked):
        if not isinstance(user, torch.TensorType):
            user = torch.tensor(user).cuda().long()

        u_emb = self.rec_model.user_encoder(user)
        i_emb = self.rec_model.item_embedding.weight

        z_u = self.act(self.proj_u(u_emb))
        z_i = self.act(self.proj_i(i_emb))

        h_u = self.binarize(z_u)
        h_i = self.binarize(z_i)

        h_u_shape = h_u.shape

        # selector = self.beta < torch.rand_like(h_u) 
        # o_u = torch.where(selector, z_u, h_u)
        # o_i = torch.where(selector, z_i, h_i)
        o_u = h_u
        o_i = h_i
        # matching = torch.mul(o_u,o_i).sum(dim=-1)
        matching = torch.matmul(o_u, o_i.T)
        matching[cliked[0],cliked[1]] = -torch.inf
        _, rec_i = torch.topk(matching, max(Ks), dim=-1)
        return rec_i



    def forward(self,user,item):
        u_emb = self.rec_model.user_encoder(user)
        i_emb = self.rec_model.item_encoder(item)

        z_u = self.act(self.proj_u(u_emb))
        z_i = self.act(self.proj_i(i_emb))

        h_u = self.binarize(z_u)
        h_i = self.binarize(z_i)

        h_u_shape = h_u.shape

        # selector = self.beta < torch.rand_like(h_u) 
        # o_u = torch.where(selector, z_u, h_u)
        # o_i = torch.where(selector, z_i, h_i)
        if self.training:
            # selector = self.beta < torch.rand([np.prod(h_u_shape[:-1])]) 
            # selector = selector.reshape(-1,1).repeat(1,h_u_shape[-1]).cuda()
            selector = self.beta < torch.rand_like(h_u) 
            o_u = torch.where(selector, z_u, h_u)
            o_i = torch.where(selector, z_i, h_i)
            # matching = torch.mul(o_u,o_i).sum(dim=-1)
            matching1 = torch.mul(u_emb,i_emb).sum(dim=-1)
            # matching2 = torch.mul(h_u, h_i).sum(dim=-1)
            matching2 = torch.mul(o_u, o_i).sum(dim=-1)
            return matching1, matching2 
            
        else:
            o_u = h_u
            o_i = h_i
            matching = torch.mul(o_u,o_i).sum(dim=-1)
            return matching


def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,all_ds_path=None,log_file=None,data_dir = None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None,need_uauc=False):
    seed=2024
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    # data_dir = "/home/zyang/LLM/MiniGPT-4/dataset/ml-100k/"
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-1m/"
    # data_dir = "/data/zyang/cf4llm/ml-1m/"
    # data_dir = "/data/zyang/datasets/ml-1m/"
    # data_dir = "/data/zyang/datasets/book/"
    # data_dir = "/home/sist/zyang/LLM/datasets/book/"
    print("data dir:", data_dir)
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values

    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1

    if all_ds_path is not None:
        all_train_data = pd.read_pickle(all_ds_path+"train_ood2.pkl")[['uid','iid','label']].values
        all_valid_data = pd.read_pickle(all_ds_path+"valid_ood2.pkl")[['uid','iid','label']].values
        all_test_data = pd.read_pickle(all_ds_path+"test_ood2.pkl")[['uid','iid','label']].values
        
        user_num = max(all_train_data[:,0].max(), all_valid_data[:,0].max(), all_test_data[:,0].max()) + 1
        item_num =  max(all_train_data[:,1].max(), all_valid_data[:,1].max(), all_test_data[:,1].max()) + 1

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass
    

    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)

    # pos number and neg number
    pos_number = train_data[:,-1].sum()
    neg_number = len(train_data[:,-1]) - pos_number
    print(f"pos_sample is {pos_number},neg_samples is {neg_number}")
    pos_weight = 1.0
    neg_weight = float(pos_number) / neg_number


    model = hashGNN(mf_config).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            users.extend(batch_data[:,0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc = 0
        if need_uauc:
            valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc = 0
        if need_uauc:
            test_uauc, _, _ = uAUC_me(users, pre, label)
        
            

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return
    
    model.beta_init()
    for epoch in range(train_config['epoch']):
        print("beta:",model.beta)
        model.train()
        epoch_loss = 0
        batch_num = 0
        for batch_id, batch_data in enumerate(train_data_loader):
            # pos_number = batch_data[:,-1].sum()
            # neg_number = len(batch_data[:,-1]) - pos_number
            # pos_weight = float(neg_number)/float(pos_number+neg_number)
            
            # neg_weight = float(pos_number)/float(pos_number+neg_number)
            # pos_weight = 1.0
            # neg_weight = float(pos_number) / neg_number

            batch_data = batch_data.cuda()
            ui_matching1,ui_matching2 = model(batch_data[:,0].long(),batch_data[:,1].long())
            pos_ui_matching1 = ui_matching1[batch_data[:,-1]==1]
            pos_ui_matching2 = ui_matching2[batch_data[:,-1]==1]
            pos_label = batch_data[:,-1][batch_data[:,-1]==1]
            
            neg_ui_matching1 = ui_matching1[batch_data[:,-1]==0]
            neg_ui_matching2 = ui_matching2[batch_data[:,-1]==0]
            neg_label = batch_data[:,-1][batch_data[:,-1]==0]
            
            # print("pos_ui_matching:",pos_ui_matching,"neg_ui_matching:",neg_ui_matching)
            # print("pos_label:",pos_label,"neg_label",neg_label)
            pos_loss = criterion(pos_ui_matching1,pos_label.float()) + 10*criterion(pos_ui_matching2,pos_label.float())
            neg_loss = criterion(neg_ui_matching1,neg_label.float()) + 10*criterion(neg_ui_matching2,neg_label.float())
            loss = pos_loss*pos_weight + neg_loss*neg_weight*10.0
           
            # loss = criterion(ui_matching, batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_num = batch_id + 1
            epoch_loss += loss.item()
        
        print(f"epoch{epoch}: loss is {epoch_loss/batch_num}")
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc = 0
            if need_uauc:
                valid_uauc, _, _ = uAUC_me(users, pre, label)

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            
            test_uauc = 0
            if need_uauc:
                test_uauc, _, _ = uAUC_me(users, pre, label)

            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
        if epoch%2 ==0:
            model.update_beta()
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()
    

# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-1]
#     dw_ = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [32,64,128,256]
#     beta = [0.02, 0.01, 0.005, 0.001]
#     try:
#         f = open("log/0130_hash_mf_lr"+str(lr_[0])+".log",'rw+')
#     except:
#         f = open("log/0130_hash_mf_lr"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 for b in beta:
#                     train_config={
#                         'lr': lr,
#                         'wd': wd,
#                         'embedding_size': embedding_size,
#                         "epoch": 5000,
#                         "eval_epoch":1,
#                         "patience":100,
#                         "batch_size":2048,
#                         "beta": b
#                     }
#                     print(train_config)
#                     run_a_trail(train_config=train_config, log_file=f, save_mode=False,need_uauc=False)
#     f.close()



# {'lr': 0.001, 'wd': 0.0001, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048},
#  {'valid_auc': 0.6760080227104877, 'valid_uauc': 0.6191863368703151, 'test_auc': 0.6482002627476354, 'test_uauc': 0.636100123360848, 'epoch': 465}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-2] #1e-2
#     dw_ = [1e-3]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [32]
#     # save_path = "/data/zyang/LLM/PretrainedModels/mf/"
#     save_path = "/data/zyang/cf4llm/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048,
#                     "beta":0.001
#                 }
#                 print(train_config)
#                 save_path += "0130hashGNN-ml-" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 # save_path = None
#                 print("save path: ", save_path)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()




#### /data/zyang/LLM/PretrainedModels/mf/best_model_d128.pth
# with prtrain version:
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-3] #1e-2
    dw_ = [1e-3]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [32]
    # save_path = "/data/zyang/LLM/PretrainedModels/mf/"
    data_dir = "/data/yanming/datasets_13_17/processed_ds/0_6/"
    save_path = "/data/yanming/datasets_13_17/processed_ds/0_6/hashGNN_models/06_05_1858"
    # try:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
    # except:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":100,
                    "batch_size":1024
                }
                print(train_config)
                # save_path = "/data/zyang/LLM/PretrainedModels/mf/0912_ml100k_oodv2_best_model_d64lr-0.001wd0.0001.pth"
                # save_path = "/data/zyang/LLM/PretrainedModels/mf/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"
                # save_path = "/data/zyang/cf4llm/0130hashGNN-ml-32lr-0.01wd0.001.pth" #"/data/zyang/cf4llm/0130hashGNN-ml-128lr-0.001wd0.001.pth"
                # save_path = "/data1/zyang/cf4reclog-cc/hash/0130hashGNN-ml-32lr-0.01wd0.001.pth"
                # if os.path.exists(save_path + "0912_ml100k_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                save_path += 'lr_'+ str(lr) + "_wd_"+str(wd) + "_ebd_" + str(embedding_size)
                #     print(save_path)
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"
                all_ds_path = None
                # all_ds_path = "/data/yanming/datasets_new/processed_ds/0_7_and_0_6/0_7/"
                run_a_trail(train_config=train_config,data_dir=data_dir, all_ds_path=all_ds_path,log_file=f, save_mode=True,save_file=save_path,need_train=True,warm_or_cold=None,need_uauc=False)
    if f is not None:
        f.close()