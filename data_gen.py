import numpy as np
from itertools import product as cartesian_prod

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
import sys

from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

np.set_printoptions(precision=2)
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def visualize(data_x, labels):
    class_0 = data_x[labels == 0]
    class_1 = data_x[labels == 1]

    plt.scatter(class_0[:, 0], class_0[:, 1])
    plt.scatter(class_1[:, 0], class_1[:, 1])
    plt.show()

class Args:
	def __init__(self):
		self.numlayer=4
		self.numnodes=50
		self.beta=3.
		self.lr=0.001       
		self.input_dim=2

args =  Args()

num_layer = args.numlayer
num_neuron = args.numnodes
beta = args.beta
lr=args.lr

saved_epochs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,64,128,256,512,1024,2048, 
                4096, 8192, 16384, 32768]
filename_suffix = str(num_layer)
filename_suffix += "_"+str(num_neuron)
filename_suffix += "_"+str(int(beta))
filename_suffix += "_"+format(lr,".1e")
print(filename_suffix)


no_of_batches=10 
weight_decay=0.0
num_hidden_nodes=[num_neuron]*num_layer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#@title Synthetic data
def set_npseed(seed):
	np.random.seed(seed)


def set_torchseed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

#Four mode classification data


def data_gen_decision_tree(num_data=1000, dim=2, seed=0, w_list=None, b_list=None, 
							vals=None, num_levels=2):
	'''
	Construct a complete decision tree with 2**num_levels-1 internal nodes, 
	e.g. num_levels=2 means there are 3 internal nodes.
	w_list, b_list is a list of size equal to num_internal_nodes, ie. weight and bias for each node 
	vals is a list of size equal to num_leaf_nodes, with values +1 or -1, ie. output of each leaf node
	'''
	# np.random.seed(6790)
	set_npseed(seed=seed)
	num_internal_nodes = 2**num_levels - 1
	num_leaf_nodes = 2**num_levels
	stats = np.zeros(num_internal_nodes+num_leaf_nodes)

	if vals is None:
		vals = np.arange(0,num_internal_nodes+num_leaf_nodes,1,dtype=np.int32)%2
		vals[:num_internal_nodes] = -99

	if w_list is None:
		w_list = np.random.standard_normal((num_internal_nodes, dim))
		w_list = w_list/np.linalg.norm(w_list, axis=1)[:, None]
		b_list = np.zeros((num_internal_nodes))

	data_x = np.random.random_sample((num_data, dim))*2 - 1.
	relevant_stats = data_x @ w_list.T + b_list
	
	print(relevant_stats[0,0], relevant_stats[0,1], relevant_stats[0,2])
	curr_index = np.zeros(shape=(num_data), dtype=int)
	
	for level in range(num_levels):
		nodes_curr_level=list(range(2**level - 1,2**(level+1)-1  ))
		for el in nodes_curr_level:
			b_list[el]=-1*np.median(relevant_stats[curr_index==el,el])
			relevant_stats[:,el] += b_list[el]
		decision_variable = np.choose(curr_index, relevant_stats.T) 
		# Go down and right if wx+b>0 down and left otherwise. 
		# i.e. 0 -> 1 if w[0]x+b[0]<0 and 0->2 otherwise
		curr_index = (curr_index+1)*2 - (1-(decision_variable > 0))

	bound_dist = np.min(np.abs(relevant_stats), axis=1)
	thres = 0
	print(relevant_stats[0,0], relevant_stats[0,1], relevant_stats[0,2])
	labels = vals[curr_index]
	data_x_pruned = data_x[bound_dist>thres]
	labels_pruned = labels[bound_dist>thres]
	relevant_stats = np.sign(data_x_pruned @ w_list.T + b_list)
	nodes_active = np.zeros((len(data_x_pruned),  num_internal_nodes+num_leaf_nodes), dtype=np.int32)
	for node in range(num_internal_nodes+num_leaf_nodes):
		if node==0:
			stats[node]=len(relevant_stats)
			nodes_active[:,0]=1
			continue
		parent = (node-1)//2
		nodes_active[:,node]=nodes_active[:,parent]
		right_child = node-(parent*2)-1 # 0 means left, 1 means right 1 has children 3,4
		if right_child==1:
			nodes_active[:,node] *= relevant_stats[:,parent]>0
		if right_child==0:
			nodes_active[:,node] *= relevant_stats[:,parent]<0		
		stats = nodes_active.sum(axis=0)
	return ((data_x_pruned, labels_pruned), (w_list, b_list, vals), stats)


# w_list = np.array([[1., 0], [0, 1], [0, 1]])
# b_list = np.array([0, 0.25, -0.25])
# vals = np.array([-99, -99, -99, 0, 1, 1, 0])
num_data = 12000
input_dim= 2
# seeds = np.random.randint(0,10000,5)
# seeds=[1387]
# # seeds = [2318]
# for seed in seeds:
# 	((data_x, labels), (w_list, b_list, vals), stats) = data_gen_decision_tree(
# 												dim=input_dim, seed=seed, num_levels=4,
# 												num_data=num_data)
# 	seed_set=seed
# w_list_old = np.array(w_list)
# b_list_old = np.array(b_list)

# num_data = len(data_x)
# num_train= num_data//2
# num_vali = num_data//4
# num_test = num_data//4
# train_data = data_x[:num_train,:]
# train_data_labels = labels[:num_train]

# vali_data = data_x[num_train:num_train+num_vali,:]
# vali_data_labels = labels[num_train:num_train+num_vali]

# test_data = data_x[num_train+num_vali :,:]
# test_data_labels = labels[num_train+num_vali :]

def check(data_x, labels, w_list, b_list, vals):
    freq = np.zeros(len(vals))
    for i, x in enumerate(data_x):
        current_index = 1
        while current_index < 16:
            output = w_list[current_index-1]@x + b_list[current_index-1]
            current_index *= 2
            if output >= 0:
                current_index += 1
        if vals[current_index-1] == 1 and labels[i] != 1:
            return False
        if vals[current_index-1] == -1 and labels[i] != 0:
            return False
        if labels[i] == 0:
            freq[current_index-1]-=1
        else:
            freq[current_index-1]+=1
    return True, freq

def get_train_loader(train_data, train_data_labels, batch_size):
        # Convert data and labels to PyTorch tensors
	train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
	train_data_labels_tensor = torch.tensor(train_data_labels, dtype=torch.long)

	# Create a TensorDataset from the data and labels tensors
	dataset = TensorDataset(train_data_tensor, train_data_labels_tensor)

	# Create a DataLoader from the dataset
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return loader
