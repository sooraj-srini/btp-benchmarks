import numpy as np
from itertools import product as cartesian_prod

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import sys
from sklearn.svm import SVC
from data_gen import set_torchseed

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


def sigmoid(u):
    u = np.maximum(u,-100)
    u = np.minimum(u,100)
    return 1/(1+np.exp(-u))





class DLGN_FC(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, num_hidden_nodes=[], beta=30, mode='pwc'):        
        super(DLGN_FC, self).__init__()
        self.num_hidden_layers = len(num_hidden_nodes)
        self.beta=beta  # Soft gating parameter
        self.mode = mode
        self.num_nodes=[input_dim]+num_hidden_nodes+[output_dim]
        self.gating_layers=nn.ModuleList()
        self.value_layers=nn.Parameter(torch.randn([1]+num_hidden_nodes)/100.)
        self.num_layer = len(num_hidden_nodes)
        self.num_hidden_nodes = num_hidden_nodes
        for i in range(self.num_hidden_layers+1):
            if i!=self.num_hidden_layers:
                temp = nn.Linear(self.num_nodes[0], self.num_nodes[i+1], bias=False)
                self.gating_layers.append(temp)

    def set_parameters_with_mask(self, to_copy, parameter_masks):
        # self and to_copy are DLGN_FC objects with same architecture
        # parameter_masks is compatible with dict(to_copy.named_parameters())
        for (name, copy_param) in to_copy.named_parameters():
            copy_param = copy_param.clone().detach()
            orig_param  = self.state_dict()[name]
            if name in parameter_masks:
                param_mask = parameter_masks[name]>0
                orig_param[param_mask] = copy_param[param_mask]
            else:
                orig_param = copy_param.data.detach()
    

                                

    def return_gating_functions(self):
        effective_weights = []
        for i in range(self.num_hidden_layers):
            curr_weight = self.gating_layers[i].weight.detach().clone()
            # curr_weight /= torch.norm(curr_weight, dim=1, keepdim=True)
            effective_weights.append(curr_weight)
        return effective_weights
        # effective_weights (and effective biases) is a list of size num_hidden_layers
                            

    def forward(self, x):
        

        for el in self.parameters():
            if el.is_cuda:
                device = torch.device('cuda:1')
            else:
                device = torch.device('cpu')
        values=[torch.ones(x.shape).to(device)]
        
        
        for i in range(self.num_hidden_layers):
            fiber = [len(x)]+[1]*self.num_layer
            fiber[i+1] = self.num_hidden_nodes[i]
            fiber = tuple(fiber)
            gate_score = torch.sigmoid( self.beta*(x@self.gating_layers[i].weight.T))#/
                #   torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T) 
            gate_score = gate_score.reshape(fiber) 
            if i==0:
                cp = gate_score
            else:
                cp = cp*gate_score 

        layers = tuple(range(1, self.num_hidden_layers + 1))
        return torch.sum(cp*self.value_layers, dim=layers)

#@title Train DLGN model
class trainDLGN:
    def __init__(self, args):
        self.lr = args.lr
        self.num_layer = args.numlayer
        self.num_neuron = args.numnodes
        self.beta = args.beta
        self.no_of_batches=32 
        self.weight_decay=0.0
        self.num_hidden_nodes=[self.num_neuron]*self.num_layer
        # self.num_hidden_nodes=[12]*4
        filename_suffix = str(self.num_layer)
        filename_suffix += "_"+str(self.num_neuron)
        filename_suffix += "_"+str(int(self.beta))
        filename_suffix += "_"+format(self.lr,".1e")
        self.filename_suffix = filename_suffix
        self.input_dim = args.input_dim
        self.saved_epochs = list(range(0,300,1)) + list(range(300,10001,50))
        self.update_value_epochs = list(range(0,10001,100))# 

    
    def train_dlgn (self, DLGN_obj, train_data_curr,vali_data_curr,test_data_curr,
                    train_labels_curr,test_labels_curr,vali_labels_curr,
                    parameter_mask=dict()):
        # DLGN_obj is the initial network
        # parameter_mask is a dictionary compatible with dict(DLGN_obj.named_parameters())
        # if a key corresponding to a named_parameter is not present it is assumed to be all ones (i.e it will be updated)
        
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        
        # Speed up of a factor of over 40 by using GPU instead of CPU
        # Final train loss of 0.02 and test acc of 74%
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        DLGN_obj.to(device)
    
        criterion = nn.CrossEntropyLoss()
    
    
    
    
        optimizer = optim.Adam(DLGN_obj.parameters(), lr=self.lr)
    
    
    
        train_data_torch = torch.Tensor(train_data_curr)
        vali_data_torch = torch.Tensor(vali_data_curr)
        test_data_torch = torch.Tensor(test_data_curr)
    
        train_labels_torch = torch.tensor(train_labels_curr, dtype=torch.int64)
        test_labels_torch = torch.tensor(test_labels_curr, dtype=torch.int64)
        vali_labels_torch = torch.tensor(vali_labels_curr, dtype=torch.int64)
    
        num_batches = self.no_of_batches
        batch_size = len(train_data_curr)//num_batches
        losses=[]
        DLGN_obj_store = []
        best_vali_error = len(vali_labels_curr)
        
    
        # print("H3")
        # print(DLGN_params)
        debug_models= []
        train_losses = []
        tepoch = tqdm(range(self.saved_epochs[-1]+1))
        for epoch in tepoch:  # loop over the dataset multiple times
            if epoch in self.update_value_epochs:
                # updating the value pathdim vector by optimising 
    
                train_preds =DLGN_obj(torch.Tensor(train_data_curr).to(device)).reshape((-1,1))
                criterion = nn.CrossEntropyLoss()
                outputs = torch.cat((-1*train_preds,train_preds), dim=1)
                targets = torch.tensor(train_labels_curr, dtype=torch.int64).to(device)
                
                train_loss = criterion(outputs, targets)
                print("Loss lefore updating value_net at epoch", epoch, " is ", train_loss)
                print("Total path abs value", torch.abs(DLGN_obj.value_layers.cpu().detach()).sum().numpy())
    
                ew = DLGN_obj.return_gating_functions()
                cp_feat = None
                for i in range(self.num_layer):
                    args = [1]*(self.num_layer + 1)
                    args[0] = -1
                    args[i+1] = self.num_neuron
                    cp_feati = sigmoid(self.beta*np.dot(train_data_curr,ew[i].cpu().T).reshape(*args))
                    if i == 0:
                        cp_feat = cp_feati
                    else:
                        cp_feat = cp_feat * cp_feati
                
                
                # cp_feat1 = sigmoid(self.beta*np.dot(train_data_curr,ew[0].cpu().T).reshape(-1,self.num_neuron,1,1,1))
                # cp_feat2 = sigmoid(self.beta*np.dot(train_data_curr,ew[1].cpu().T).reshape(-1,1,self.num_neuron,1,1))
                # cp_feat3 = sigmoid(self.beta*np.dot(train_data_curr,ew[2].cpu().T).reshape(-1,1,1,self.num_neuron,1))
                # cp_feat4 = sigmoid(self.beta*np.dot(train_data_curr,ew[3].cpu().T).reshape(-1,1,1,1,self.num_neuron))
                # cp_feat = cp_feat1 * cp_feat2 * cp_feat3 * cp_feat4
                cp_feat_vec = cp_feat.reshape((len(cp_feat),-1))
    
                clf = LogisticRegression(C=0.03, fit_intercept=False,max_iter=1000, penalty="l1", solver='liblinear')
                clf.fit(2*cp_feat_vec, train_labels_curr)
                shape_args = [self.num_neuron]*(self.num_layer + 1)
                shape_args[0] = 1
                value_wts  = clf.decision_function(np.eye(self.num_neuron**self.num_layer)).reshape(*shape_args)
                # value_wts  = clf.decision_function(np.eye(self.num_neuron**self.num_layer)).reshape(1,self.num_neuron,self.num_neuron,self.num_neuron, self.num_neuron)
                
                A= DLGN_obj.value_layers.detach()
                A[:] = torch.Tensor(value_wts)
    
                train_preds =DLGN_obj(torch.Tensor(train_data_curr).to(device)).reshape((-1,1))
                criterion = nn.CrossEntropyLoss()
                outputs = torch.cat((-1*train_preds,train_preds), dim=1)
                targets = torch.tensor(train_labels_curr, dtype=torch.int64).to(device)
                train_loss = criterion(outputs, targets)
                print("Loss after updating value_net at epoch", epoch, " is ", train_loss)            
                print("Total path abs value", torch.abs(DLGN_obj.value_layers.cpu().detach()).sum().numpy())
            if epoch in self.saved_epochs:
                DLGN_obj_copy = deepcopy(DLGN_obj)
                DLGN_obj_copy.to(torch.device('cpu'))
                DLGN_obj_store.append(DLGN_obj_copy)
            
            for batch_start in range(0,len(train_data_curr),batch_size):
                if (batch_start+batch_size)>len(train_data_curr):
                    break
                optimizer.zero_grad()
                inputs = train_data_torch[batch_start:batch_start+batch_size]
                targets = train_labels_torch[batch_start:batch_start+batch_size].reshape(batch_size)
                criterion = nn.CrossEntropyLoss()
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = DLGN_obj(inputs).reshape(-1,1)
                # preds_clone = preds.detach().clone().cpu().numpy()[:,0]
                # targets_clone = targets.detach().clone().cpu().numpy()
                # coeff = (0.5-targets_clone)/(sigmoid(2*preds_clone)-targets_clone)
                # print(coeff.shape)
                
                # print(coeff.min())
                # print(coeff.mean())
                # print(coeff.max())
                outputs = torch.cat((-1*preds, preds), dim=1)
                loss = criterion(outputs, targets)
                # loss = loss*torch.tensor(coeff, device=device)    
                # loss = loss.mean()        
                loss.backward()
                for name,param in DLGN_obj.named_parameters():
                    if "val" in name:
                        param.grad *= 0.0
                    if "gat" in name:
                        param.grad *= 1.0
                optimizer.step()
    
            train_preds =DLGN_obj(torch.Tensor(train_data_curr).to(device)).reshape(-1,1)
            criterion = nn.CrossEntropyLoss()
            outputs = torch.cat((-1*train_preds,train_preds), dim=1)
            targets = torch.tensor(train_labels_curr, dtype=torch.int64).to(device)
            train_loss = criterion(outputs, targets)
            if epoch%25 == 0:
                print("Loss after updating at epoch ", epoch, " is ", train_loss)
                test_preds =DLGN_obj(test_data_torch.to(device)).reshape(-1,1)
                test_preds = test_preds.detach().cpu().numpy()
                print("Test error=",np.sum(test_labels_curr != (np.sign(test_preds[:,0])+1)//2 ))
            if train_loss < 0.005:
                break
            if np.isnan(train_loss.detach().cpu().numpy()):
                break
    
            losses.append(train_loss.cpu().detach().clone().numpy())
            inputs = vali_data_torch.to(device)
            targets = vali_labels_torch.to(device)
            preds =DLGN_obj(inputs).reshape(-1,1)
            vali_preds = torch.cat((-1*preds, preds), dim=1)
            vali_preds = torch.argmax(vali_preds, dim=1)
            vali_error= torch.sum(targets!=vali_preds)
            if vali_error < best_vali_error:
                DLGN_obj_return = deepcopy(DLGN_obj)
                best_vali_error = vali_error
        plt.figure()
        plt.title("DLGN loss vs epoch")
        plt.plot(losses)
        if not os.path.exists('figures'):
            os.mkdir('figures')
    
        filename = 'figures/'+self.filename_suffix +'.pdf'
        plt.savefig(filename)
        DLGN_obj_return.to(torch.device('cpu'))
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        return train_losses, DLGN_obj_return, DLGN_obj_store, losses, debug_models
    

    def train(self,train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
        set_torchseed(6675)
        # set_torchseed(5449)
        DLGN_init= DLGN_FC(input_dim=self.input_dim, output_dim=1, num_hidden_nodes=self.num_hidden_nodes, beta=self.beta)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_parameter_masks=dict()
        for name,parameter in DLGN_init.named_parameters():
            if name[:5]=="value_"[:5]:
                train_parameter_masks[name]=torch.ones_like(parameter) # Updating all value network layers
            if name[:5]=="gating_"[:5]:
                train_parameter_masks[name]=torch.ones_like(parameter)
            train_parameter_masks[name].to(device)

        set_torchseed(5000)
        train_losses, DLGN_obj_final, DLGN_obj_store, _, _ = self.train_dlgn(train_data_curr=train_data,
                                                    vali_data_curr=vali_data,
                                                    test_data_curr=test_data,
                                                    train_labels_curr=train_data_labels,
                                                    vali_labels_curr=vali_data_labels,
                                                    test_labels_curr=test_data_labels,
                                                    DLGN_obj=deepcopy(DLGN_init),
                                                    parameter_mask=train_parameter_masks)
        torch.cuda.empty_cache() 

        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        # print(len(DLGN_obj_store))
        # print("Hi")
        device=torch.device('cpu')
        train_outputs_values =DLGN_obj_final(torch.Tensor(train_data).to(device))
        train_preds = train_outputs_values[-1]
        criterion = nn.CrossEntropyLoss()
        outputs = torch.cat((-1*train_preds,train_preds), dim=1)
        targets = torch.tensor(train_data_labels, dtype=torch.int64)
        train_loss = criterion(outputs, targets)
        train_preds = train_preds.detach().numpy()
        filename = 'outputs/'+self.filename_suffix+'.txt'
        original_stdout = sys.stdout
        # with open(filename,'w') as f:
            # sys.stdout = f
        print("Setup:")
        print("Num neurons : ", DLGN_obj_final.num_nodes)
        print(" Beta :", DLGN_obj_final.beta)
        print(" lr :", self.lr)
        print("=======================")
        print(train_losses)
        print("==========Best validated model=============")
        print("Train error=",np.sum(train_data_labels != (np.sign(train_preds[:,0])+1)//2 ))
        print("Train loss = ", train_loss)
        print("Num_train_data=",len(train_data_labels))
        sys.stdout = original_stdout


        test_outputs_values, test_outputs_gate_scores =DLGN_obj_final(torch.Tensor(test_data))
        test_preds = test_outputs_values[-1]
        test_preds = test_preds.detach().numpy()
        filename = 'outputs/'+self.filename_suffix+'.txt'
        original_stdout = sys.stdout
        # with open(filename,'a') as f:
            # sys.stdout = f
        test_error = np.sum(test_data_labels != (np.sign(test_preds[:,0])+1)//2 )
        test_error_acc = 100 - (test_error/len(test_data_labels))*100
        print("Test error=",np.sum(test_data_labels != (np.sign(test_preds[:,0])+1)//2 ))
        print("Num_test_data=",len(test_data_labels))
        print("Test accuracy=", test_error_acc)
        sys.stdout = original_stdout

        # w_list = np.concatenate((w_list_old,-w_list_old),axis=0)

        # effective_weights, effective_biases = DLGN_obj_store[0].return_gating_functions()
        # wts_list_init=[]
        # for layer in range(0,len(effective_weights)):
        #     wts =  np.array(effective_weights[layer].data.detach().numpy())
        #     wts /= np.linalg.norm(wts, axis=1)[:,None]
        #     wts_list_init.append(wts)
        # wts_list_init = np.concatenate(wts_list_init)


        # effective_weights, effective_biases = DLGN_obj_final.return_gating_functions()

        # wts_list=[]
        # for layer in range(len(effective_weights)):
        #     wts =  np.array(effective_weights[layer].data.detach().numpy())
        #     wts /= np.linalg.norm(wts, axis=1)[:,None]
        #     wts_list.append(wts)
        # wts_list = np.concatenate(wts_list)

        # pd0 =  pairwise_distances(w_list,wts_list_init)
        # pd1 =  pairwise_distances(w_list,wts_list)

