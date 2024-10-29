import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GATConv
import numpy as np
import csv
import random
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
import time

from pygsp import graphs, filters, reduction
from scipy.sparse import csr_matrix
from coarsening_utils import coarsen
from sklearn.neighbors import NearestNeighbors

def check_self_loop(g):
    sources = g.edges()[0].cpu().numpy()
    targets = g.edges()[1].cpu().numpy()
    n = sum(sources == targets)
    return n>0

def dgl2pyg(dgl_g):
    sources = dgl_g.edges()[0].cpu().numpy()
    targets = dgl_g.edges()[1].cpu().numpy()
    weights = np.ones(sources.shape[0])
    n_nodes = dgl_g.ndata['x'].size()[0]
    A = csr_matrix((weights, (sources, targets)), shape=(n_nodes, n_nodes))
    pyg_g = graphs.Graph(A)
    return pyg_g

def pyg2dgl(pyg_g):
    sources = pyg_g.W.tocoo().row
    targets = pyg_g.W.tocoo().col
    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(pyg_g.N)
    dgl_g.add_edges(sources, targets)
    return dgl_g

def csr2tensor(Acsr):
    Acoo = Acsr.tocoo()
    return torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.FloatTensor(Acoo.data.astype(np.int32))).cuda()

def combine_graph(g, Gc=None, C=None, c2n=None, n2c=None, device=torch.device('cuda:0')):
    if len(g.ndata['y'].size()) == 1:
        g.ndata['y'] = F.one_hot(g.ndata['y']).float()
    # train_val_mask = torch.logical_or(g.ndata['train_mask'], g.ndata['valid_mask'])

    if Gc == None:
        n_nodes = g.ndata['node_idxs'].size()[0]
        c2n = dict((i,set([g.ndata['node_idxs'][i].item()])) for i in range(n_nodes))
        # n2c = dict((g.ndata['node_idxs'][i].item(), i) for i in range(n_nodes))
        max_node_idx = torch.max(g.ndata['node_idxs']).item()
        n2c = torch.tensor([-1 for i in range(max_node_idx+1)])
        n2c[g.ndata['node_idxs']] = torch.tensor([i for i in range(n_nodes)])
        return g, c2n, n2c

    num_old_nodes, num_old_clusters = C.shape[1], C.shape[0]
    # num_new_nodes = g.ndata['num_new_nodes'][0]
    # new_node_idxs = torch.tensor([i for i in range(num_new_nodes)])
    new_node_idxs = g.ndata['new_nodes_mask'].nonzero().squeeze()
    new_node_features = g.ndata['x'][new_node_idxs].float()
    new_node_labels = g.ndata['y'][new_node_idxs]
    num_new_nodes = new_node_features.size()[0]

    new_train_mask = g.ndata['train_mask'][new_node_idxs]
    new_valid_mask = g.ndata['valid_mask'][new_node_idxs]
    # new_test_mask = g.ndata['valid_mask'][new_node_idxs]

    Gc.add_nodes(num_new_nodes, {'x':new_node_features, 'y':new_node_labels,
                                 'train_mask':new_train_mask, 'valid_mask':new_valid_mask})

    max_new_node_idx = torch.max(g.ndata['node_idxs'][new_node_idxs]).item()
    # print ('max_new_node_idx',max_new_node_idx)
    if n2c.size()[0] < max_new_node_idx+1:
        n2c = torch.cat([n2c, torch.tensor([-1 for i in range(max_new_node_idx+1-n2c.size()[0])]).to(device)], 0)
    n2c[g.ndata['node_idxs'][new_node_idxs]] = torch.tensor([i+len(c2n) for i in range(new_node_idxs.shape[0])]).to(device)
    # print ((n2c>=0).nonzero())
    rows = g.ndata['node_idxs'][g.edges()[0]].to(device)
    cols = g.ndata['node_idxs'][g.edges()[1]].to(device)

    # print ((n2c[rows]>=0).nonzero().squeeze().tolist())
    valid_row_idx = set((n2c[rows]>=0).nonzero().squeeze().tolist())
    valid_col_idx = set((n2c[cols]>=0).nonzero().squeeze().tolist())

    valid_edge_idx = torch.tensor(list(valid_col_idx.intersection(valid_row_idx))).to(device)


    g.ndata['node_idxs'][new_node_idxs]
    Gc.add_edges(n2c[rows][valid_edge_idx], n2c[cols][valid_edge_idx])

    c2n_extend = dict((n2c[g.ndata['node_idxs'][idx]].item(),set([g.ndata['node_idxs'][idx].item()])) for i,idx in enumerate(new_node_idxs))
    c2n.update(c2n_extend)

    #  ('combined_g:', Gc)
    return Gc, c2n, n2c

def graph_coarsening(g, node_hidden_features, c2n, n2c, train_ratio, reduction, replay_nodes, device=torch.device('cuda:0')):
    features = g.ndata['x'].float()
    labels = g.ndata['y']
    g_pyg = dgl2pyg(g)
    start_time = time.time()
    C, Gc, Call, Gall = coarsen(G_orig=g_pyg, G=g_pyg, r=reduction, replay_nodes=replay_nodes, feature=node_hidden_features)
    end_time = time.time()
    C_tensor = csr2tensor(C)
    coarsened_features = C_tensor@features
    g_dgl = pyg2dgl(Gc).to(device)
    # print ('after coarsening:')
    # print (g_dgl)

    g_dgl.ndata['x'] = coarsened_features
    n2c_new = torch.tensor([-1 for el in range(n2c.size()[0])]).to(device)
    c2n_new = dict((el,set()) for el in range(C.shape[0]))
    for i in range(C.shape[0]):
        cluster_i = [x for x in C.getrow(i).nonzero()[1]]
        for super_node in cluster_i:
            nodes = c2n[super_node]
            c2n_new[i].update(nodes)
            n2c_new[torch.tensor(list(nodes))] = torch.tensor([i for _ in range(len(nodes))]).to(device)

    train_val_mask = torch.logical_or(g.ndata['train_mask'], g.ndata['valid_mask'])
    coarsened_mask = C_tensor@(train_val_mask.float())
    coarsened_mask = coarsened_mask>0
    coarsened_labels = C_tensor@(labels.t()*train_val_mask).t()
    g_dgl.ndata['y'] = coarsened_labels

    mask_idxs = coarsened_mask.nonzero().squeeze().tolist()
    random.shuffle(mask_idxs)
    num_training_sample = int(len(mask_idxs)*train_ratio)
    coarsened_train_mask = torch.zeros(len(c2n_new)).bool().to(device)
    coarsened_valid_mask = torch.zeros(len(c2n_new)).bool().to(device)
    coarsened_train_mask[torch.tensor(mask_idxs[:num_training_sample]).long()] = True
    coarsened_valid_mask[torch.tensor(mask_idxs[num_training_sample:]).long()] = True
    g_dgl.ndata['train_mask'] = coarsened_train_mask
    g_dgl.ndata['valid_mask'] = coarsened_valid_mask

    return g_dgl, C, c2n_new, n2c_new


def evaluate(net, g, features, labels, test_mask):
    g.add_edges(g.nodes(),g.nodes())
    y_pred = net(g, features)[test_mask].cpu()
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = labels[test_mask].cpu()
    bac = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return bac, f1, acc

def record_results(f, ave_test_acc_raw, ave_test_bac_raw, ave_test_f1_raw):
    writer = csv.writer(f)

    writer.writerow('acc')
    for i in range(len(ave_test_acc_raw)):
        row = []
        for j in range(len(ave_test_acc_raw[i])):
            std = np.std(ave_test_acc_raw[i][j])
            avg = np.mean(ave_test_acc_raw[i][j])
            content = "{:.2f}".format(avg*100) + u"\u00B1" + "{:.2f}".format(std*100) + '%'
            row.append(content)
        writer.writerow(row)

    writer.writerow('bac')
    for i in range(len(ave_test_bac_raw)):
        row = []
        for j in range(len(ave_test_bac_raw[i])):
            std = np.std(ave_test_bac_raw[i][j])
            avg = np.mean(ave_test_bac_raw[i][j])
            content = "{:.2f}".format(avg*100) + u"\u00B1" + "{:.2f}".format(std*100) + '%'
            row.append(content)
        writer.writerow(row)

    writer.writerow('f1')
    for i in range(len(ave_test_f1_raw)):
        row = []
        for j in range(len(ave_test_f1_raw[i])):
            std = np.std(ave_test_f1_raw[i][j])
            avg = np.mean(ave_test_f1_raw[i][j])
            content = "{:.2f}".format(avg*100) + u"\u00B1" + "{:.2f}".format(std*100) + '%'
            row.append(content)
        writer.writerow(row)

    f.close()
