import argparse
from dgl.data import register_data_args
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
# import pickle
# import random
import numpy as np
# from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
# import csv
import quadprog
import torch.optim as optim
import copy
import random
import collections

class DYGRA_reservior(torch.nn.Module):
    def __init__(self,
                 model, opt, num_class, buffer_size,
                 args):
        super(DYGRA_reservior, self).__init__()

        # self.task_manager = task_manager
        # setup network
        self.net = model
        # setup optimizer
        self.opt = opt
        # setup memories
        self.current_task = 0
        self.num_samples = 0
        self.buffer_size = buffer_size # memory size
        self.er_buffer = []

    def update_er_buffer(self, g):
        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            self.num_samples += 1
            if len(self.er_buffer) < self.buffer_size:
                self.er_buffer.append(g.ndata['node_idxs'][node])
            else:
                rand_idx = random.randint(0, self.num_samples-1)
                if rand_idx < self.buffer_size:
                    self.er_buffer[rand_idx] = g.ndata['node_idxs'][node]
        return self.er_buffer

    def forward(self, g, features):
        # print (self.net.return_hidden)
        output = self.net(g, features)
        return output

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'],1).indices
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        self.net.train()
        self.net.zero_grad()

        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        loss.backward()
        self.opt.step()

class DYGRA_meanfeature(torch.nn.Module):
    def __init__(self,
                 model, opt, num_class, buffer_size,
                 args):
        super(DYGRA_meanfeature, self).__init__()

        # self.task_manager = task_manager
        # setup network
        self.net = model
        # setup optimizer
        self.opt = opt
        # setup memories
        self.current_task = 0
        self.num_samples = 0
        self.num_class = num_class
        self.buffer_size_per_class = buffer_size // num_class # memory size
        self.moving_avg = [-1 for i in range(num_class)]
        self.buffer_size = buffer_size # memory size
        self.er_buffer = [[] for _ in range(num_class)]
        self.appeared_samples = [0 for i in range(num_class)]

    def update_er_buffer(self, g, t):
        g = copy.deepcopy(g)
        g.add_edges(g.nodes(), g.nodes())
        self.net.return_hidden_layer = True

        hidden_features = self.net(g, g.ndata['x'])
        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            node_hidden_feature = hidden_features[node]
            n = self.appeared_samples[node_label]
            if n == 0:
                self.moving_avg[node_label] = node_hidden_feature
            else:
                self.moving_avg[node_label] = self.moving_avg[node_label]/(n+1)*n + node_hidden_feature/(n+1)
            d = torch.cdist(self.moving_avg[node_label], node_hidden_feature)
            # print (d)
            if len(self.er_buffer[node_label]) < self.buffer_size_per_class:
                self.er_buffer[node_label].append((node_feature, d, node))
            else:
                max_d = max(self.er_buffer[node_label], key=lambda x: x[1])
                # print (max_d)
                max_d_idx = [x[1] for x in self.er_buffer[node_label]].index(max_d[1])
                if d < max_d[1]:
                    self.er_buffer[node_label][max_d_idx] = (node_feature, d, node)
            self.appeared_samples[node_label] += 1
        self.net.return_hidden_layer = False

        er_nodes = []
        for i in range(len(self.er_buffer)):
            er_nodes.extend((x[2] for x in self.er_buffer[i]))
        return er_nodes

    def forward(self, g, features):
        output = self.net(g, features)
        return output

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'],1).indices
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        self.net.train()
        self.net.zero_grad()

        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        loss.backward()
        self.opt.step()

class DYGRA_ringbuffer(nn.Module):
    def __init__(self,
                model,
                opt, num_class, buffer_size,
                args,  device=torch.device('cuda:0')):
        super(DYGRA_ringbuffer, self).__init__()
        self.net = model
        self.opt = opt
        self.buffer_size_per_class = buffer_size // num_class # memory size
        self.er_buffer = [collections.deque([], self.buffer_size_per_class) for _ in range(num_class)]
        self.num_samples = 0
        self.num_class = num_class
        # self.alpha = args.alpha
        self.device = device

    def forward(self, g, features):
        g = copy.deepcopy(g)
        g.add_edges(g.nodes(), g.nodes())
        output = self.net(g, features)
        return output

    def update_er_buffer(self, g, t):
        # g = g_list[t]
        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            self.er_buffer[node_label].appendleft((node_feature, node))
        er_nodes = []
        for i in range(len(self.er_buffer)):
            er_nodes.extend([x[1] for x in self.er_buffer[i]])
        return er_nodes
    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = g.ndata['y']
        train_mask = g.ndata['train_mask']

        self.net.train()
        self.net.zero_grad()
        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        if t>0:
            g_rp = dgl.DGLGraph().to(self.device)

            for node_label in range(self.num_class):
                if len(self.er_buffer[node_label]) > 0:
                    x = torch.cat(list([z[0] for z in self.er_buffer[node_label]]))
                    label = torch.tensor([node_label for _ in range(len(self.er_buffer[node_label]))]).to(self.device)
                    g_rp.add_nodes(len(self.er_buffer[node_label]), {'x': x, 'y': label})

            g_rp.add_edges(g_rp.nodes(), g_rp.nodes())
            output = self.net(g_rp, g_rp.ndata['x'])
            output = F.log_softmax(output, 1)

            loss = loss*0.5 + (1-0.5)*loss_func(output, g_rp.ndata['y'])
        loss.backward()
        self.opt.step()
