import argparse
from dgl.data import register_data_args
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import pickle
import random
import numpy as np
# from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
import csv


dataset = 'ACM'
datapath = '../data/ACM/'
train_ratio, valid_ratio, test_ratio = 0.3, 0.2, 0.5

with open(datapath+'statistics', 'rb') as file:
    num_tasks, num_class = pickle.load(file)

for run in range(10):
    train_node_idxs_set, valid_node_idxs_set, test_node_idxs_set = set(),set(),set()
    masks_supgraphs_list = []
    for time_slot in range(num_tasks):
        with open(datapath+f'sub_graph_{time_slot}_by_edges', 'rb') as file:
            g = pickle.load(file)
        n_nodes = g.num_nodes()
        excluded_class = random.randint(0,num_class-1)
        print (excluded_class)
        selected_class_mask = (g.ndata['y']-excluded_class).bool()
#         new_nodes_mask = g.ndata['new_nodes_mask']
        new_nodes_mask = torch.logical_and(selected_class_mask,g.ndata['new_nodes_mask'])
#         print (sum(new_nodes_mask))
        n_new_nodes = sum(new_nodes_mask)
        new_node_idxs = new_nodes_mask.nonzero()
        n_new_nodes = len(new_node_idxs)
        shuffled_ind = np.array([i for i in range(n_new_nodes)])
        random.shuffle(shuffled_ind)
        ratio_train, ratio_valid, ratio_test = train_ratio, valid_ratio, test_ratio
        # mask_file = f'data/masks_{args.Dataset}_t_{time_slot}_run_{}'
        n_train, n_valid, n_test = int(ratio_train*n_new_nodes), int(ratio_valid*n_new_nodes), int(ratio_test*n_new_nodes)
        ind_train, ind_valid, ind_test = \
        shuffled_ind[:n_train], shuffled_ind[n_train:n_train+n_valid], shuffled_ind[n_train+n_valid:n_new_nodes]
        ind_train, ind_valid, ind_test = new_node_idxs[ind_train],new_node_idxs[ind_valid],new_node_idxs[ind_test]

        train_mask, valid_mask, test_mask = torch.tensor([False for i in range(n_nodes)]), torch.tensor([False for i in range(n_nodes)]), torch.tensor([False for i in range(n_nodes)])
        train_mask[ind_train] = True
        valid_mask[ind_valid] = True
        test_mask[ind_test] = True


        masks_supgraphs_list.append((train_mask, valid_mask, test_mask))

        train_node_idxs = g.ndata['node_idxs'][train_mask]
        valid_node_idxs = g.ndata['node_idxs'][valid_mask]
        test_node_idxs = g.ndata['node_idxs'][test_mask]

        train_node_idxs_set.update(train_node_idxs.tolist())
        valid_node_idxs_set.update(valid_node_idxs.tolist())
        test_node_idxs_set.update(test_node_idxs.tolist())

    # run = 0
    with open(datapath+f'mask_seed_{run}', 'wb') as file:
      pickle.dump(masks_supgraphs_list, file)
