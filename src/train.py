import argparse
import os
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
import copy
import os
import time

# from dgl.nn import GATConv
from models import *
from utils import *
from gcl_methods import *
# from models_twp import get_model

def run(args):
    use_gpu = (args.gpu == 0)
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    num_runs = args.num_runs
    datapath = f'../data/{args.Dataset}/'
    with open(datapath+'statistics', 'rb') as file:
        num_task, num_class = pickle.load(file)
    ave_test_acc, ave_test_bac, ave_test_f1 = 0.,0.,0.
    ave_test_acc_list = [0. for i in range(num_task)]
    ave_test_bac_list = [0. for i in range(num_task)]
    ave_test_f1_list = [0. for i in range(num_task)]

    ave_test_acc_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]
    ave_test_bac_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]
    ave_test_f1_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]

    for run in range(num_runs):
    # for run in range(1):
        # num_class = 0
        g_list = []
        with open(datapath+f'mask_seed_{run}', 'rb') as file:
            masks_supgraphs_list = pickle.load(file)
        for time_slot in range(num_task):
            with open(datapath+f'sub_graph_{time_slot}_by_edges', 'rb') as file:
                g = pickle.load(file)
            n_nodes = g.num_nodes()
            train_mask, valid_mask, test_mask = masks_supgraphs_list[time_slot]
            g.ndata['train_mask'] = train_mask
            g.ndata['valid_mask'] = valid_mask
            g.ndata['test_mask'] = test_mask

            if use_gpu:
                g = g.to(device)
            g_list.append(g)
        input_dim = g.ndata['x'].size()[1]

        if args.method == "DYGRA":
            gcl_method = DYGRA_reservior
        else if args.method == "DYGRA_meanfeature":
            gcl_method = DYGRA_meanfeature
        else if args.method == "DYGRA_ringbuffer":
            gcl_method = DYGRA_ringbuffer

        test_acc_list, test_bac_list, test_f1_list = [],[],[]
        num_test_list = []
        best_model_path = 'best_model_stat_dict'
        if args.gnn == "GAT":
            net = GAT(input_dim, hidden_dim, num_class, num_heads)
        elif args.gnn == "GCN":
            net = GCN(input_dim, hidden_dim, num_class)
        elif args.gnn == "GIN":
            net = GIN(input_dim, hidden_dim, num_class)
        if use_gpu:
            net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        buffer_size = args.buffer_size
        gcl = gcl_method(net, optimizer, num_class, buffer_size, args)
        combined_g_list = []
        for train_slot in range(num_task):
            print ('train_slot:', train_slot)
            g = g_list[train_slot]
            if args.nfp:
                er_buffer = gcl.update_er_buffer(g)
            else:
                er_buffer = []
            if train_slot == 0:
                combined_g, c2n, n2c = combine_graph(g)
            else:
                combined_g, c2n, n2c = combine_graph(g, coarsened_g, C, c2n, n2c)

            replay_nodes = n2c[torch.tensor(er_buffer)]
            combined_g_list.append(combined_g)
            # continue
            features = combined_g.ndata['x']
            labels = torch.max(combined_g.ndata['y'],1).indices
            train_mask = combined_g.ndata['train_mask']
            valid_mask = combined_g.ndata['valid_mask']

            loss_func = nn.CrossEntropyLoss()
            all_logits=[]
            best_bac = 0


            for epoch in range(args.num_epochs):
                gcl.observe(combined_g_list, train_slot, loss_func)
                valid_bac, valid_f1, valid_acc = evaluate(gcl, copy.deepcopy(combined_g), features, labels, valid_mask)
                if valid_bac > best_bac:
                    torch.save(net.state_dict(), best_model_path)

            gcl.net.load_state_dict(torch.load(best_model_path))
            # if args.method == "simple_reg":
            #     gcl.last_model.load_state_dict(torch.load(best_model_path))
    #         test_acc_list = []
    #         num_test_list = []
            gcl.net.return_hidden = True
            combined_g_copy = copy.deepcopy(combined_g)
            combined_g_copy.add_edges(combined_g_copy.nodes(), combined_g_copy.nodes())
            node_hidden_features = gcl.net(combined_g_copy, combined_g_copy.ndata['x']).detach()
            gcl.net.return_hidden = False
            coarsened_g, C, c2n, n2c = graph_coarsening(combined_g, node_hidden_features, c2n, n2c, 0.6, args.reduction_rate, replay_nodes)

            avg_test_acc, avg_test_bac, avg_test_f1 = 0., 0., 0.
            for test_slot in range(num_task):
                # print ('test slot:', test_slot)
                g = g_list[test_slot]
                features = g.ndata['x']
                if test_slot <= train_slot:
                    labels = torch.max(g.ndata['y'],1).indices
                else:
                    labels = g.ndata['y']
                test_mask = g.ndata['test_mask']
                test_bac, test_f1, test_acc = evaluate(gcl, copy.deepcopy(g), features, labels, test_mask)
                avg_test_acc += test_acc/num_task
                avg_test_bac += test_bac/num_task
                avg_test_f1 += test_f1/num_task
                ave_test_acc_raw[train_slot][test_slot][run] = test_acc
                ave_test_bac_raw[train_slot][test_slot][run] = test_bac
                ave_test_f1_raw[train_slot][test_slot][run] = test_f1
            ave_test_acc_raw[train_slot][-1][run] = avg_test_acc
            ave_test_bac_raw[train_slot][-1][run] = avg_test_bac
            ave_test_f1_raw[train_slot][-1][run] = avg_test_f1
            print (avg_test_f1, avg_test_bac)


        for i in range(num_task):
            ave_test_acc_raw[-1][i][run] = max([ave_test_acc_raw[j][i][run] for j in range(num_task)]) - ave_test_acc_raw[-2][i][run]
            ave_test_bac_raw[-1][i][run] = max([ave_test_bac_raw[j][i][run] for j in range(num_task)]) - ave_test_bac_raw[-2][i][run]
            ave_test_f1_raw[-1][i][run] = max([ave_test_f1_raw[j][i][run] for j in range(num_task)]) - ave_test_f1_raw[-2][i][run]

        ave_test_acc_raw[-1][-1][run] = sum([ave_test_acc_raw[-1][i][run] for i in range(num_task)])/num_task
        ave_test_bac_raw[-1][-1][run] = sum([ave_test_bac_raw[-1][i][run] for i in range(num_task)])/num_task
        ave_test_f1_raw[-1][-1][run] = sum([ave_test_f1_raw[-1][i][run] for i in range(num_task)])/num_task

    results_each_run = (ave_test_acc_raw[-2][-1], ave_test_bac_raw[-2][-1], ave_test_f1_raw[-2][-1])
    pickle.dump(results_each_run, open(f'results/{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}', 'wb'))


    result_path = f'../results/{args.Dataset}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results_each_run = (ave_test_acc_raw[-2][-1], ave_test_bac_raw[-2][-1], ave_test_f1_raw[-2][-1])
    pickle.dump(results_each_run, open(f'results/{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}', 'wb'))

    f = open(result_path+f'{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}.csv', 'w')
    record_results(f, ave_test_acc_raw, ave_test_bac_raw, ave_test_f1_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCL')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Dataset", type=str, default='kindle',
                        help=False)
    parser.add_argument("--num_runs", type=int, default=10,
                        help="number of runs, default = 10")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="number of training epochs, default = 50")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--gnn', type=str, default='GCN',
                        help="basemodel")
    parser.add_argument('--method', type=str, default="DYGRA")
    parser.add_argument("--reduction_rate", type=float, default=0.5,
                    help="reduction rate")
    parser.add_argument("--nfp", type=bool, default=True,
                    help="node fidality preservation")

    # parameters for GAT model
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=48,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--buffer_size', type=int, default=200)


    args = parser.parse_args()
    # set_seed(args)
    run(args)
