import json
import csv
from collections import defaultdict
import pickle
import torch
import dgl
import os
import numpy as np

datapath = f'../data/kindle/'
with open(datapath+'statistics', 'rb') as file:
    num_task, num_class = pickle.load(file)

from collections import deque

def dfs(g, u, visited):
    q = deque([u])
    subnodes = []
    count = 0
    while q:
        v = q.popleft()
        if v in visited:
            continue
        visited.add(v)
        subnodes.append(v)
        count += 1
        node_list = g.edges()[1][(g.edges()[0] == v).nonzero().view(-1)].tolist()
        node_list += g.edges()[0][(g.edges()[1] == v).nonzero().view(-1)].tolist()
        q.extend((x for x in node_list if x not in visited))
    return count, subnodes

def get_largest_cluster(g):
    count_list = []
    visited = set()
    for u in g.nodes().tolist():
        if u not in visited:
#             print (len(visited))
            count_list.append(dfs(g,u, visited))
    count_list.sort(reverse=True)
    return count_list[0][1]

sub_g_list = []
for time_slot in range(num_task):
    with open(datapath+f'graph_{time_slot}_by_edges', 'rb') as file:
        g = pickle.load(file)
        subnodes = get_largest_cluster(g)
        sub_g = g.subgraph(subnodes)
        print (sub_g)
        sub_g_list.append(sub_g)

appeared_nodes = set()
for i in range(num_task):
    appeared_nodes.update(sub_g_list[i].ndata['node_idxs'].tolist())

g_list = []
for time_slot in range(num_task):
    with open(datapath+f'graph_{time_slot}_by_edges', 'rb') as file:
        g = pickle.load(file)
        g_list.append(g)

prev_idxs = set()

for t in range(num_task):
    curr_idxs = sub_g_list[t].ndata['node_idxs'].tolist()
    new_nodes = [i for i in range(len(curr_idxs)) if sub_g_list[t].ndata['node_idxs'][i].detach().item() not in prev_idxs]
    new_nodes_mask = np.zeros(len(curr_idxs))
    new_nodes_mask[new_nodes] = 1
    sub_g_list[t].ndata['new_nodes_mask'] = torch.tensor(new_nodes_mask).int()

    prev_idxs.update(node_idx for node_idx in sub_g_list[t].ndata['node_idxs'].tolist())

for i in range(num_task):
    with open(datapath+f'sub_graph_{i}_by_edges', 'wb') as file:
      pickle.dump(sub_g_list[i], file)
