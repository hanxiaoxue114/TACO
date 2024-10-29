import dgl
import torch
import os
import json
import pickle
from urllib.request import urlopen
from nltk.tokenize import word_tokenize
from collections import defaultdict


datafile = '../raw-data/ACM_processed.pickle'
### load the meta data
with open(datafile, 'rb') as handle:
    data = pickle.load(handle)

venue_dict = defaultdict(int)
for i, (id, item) in enumerate(data.items()):
    if 'venue' in item:
        venue_dict[item['venue']] += 1

venue_list = []
for venue in venue_dict:
    venue_list.append((venue_dict[venue], venue))
venue_list = sorted(venue_list, reverse=True)
venue_set = set([v[1] for v in venue_list])


from venue_classes import iss, sp, am, ai
iss, sp, am, ai = set(iss), set(sp), set(am), set(ai)

def get_label(venue):
    if venue in iss: return 0
    if venue in sp: return 1
    if venue in am: return 2
    if venue in ai: return 3
    return -1

start_year = 1995
end_year = 2014
num_task = 10
num_papers_year = [0 for i in range(num_task)]

token_dic = defaultdict(int)
for i, (id, item) in enumerate(data.items()):
    if 'venue' in item and 'year' in item and 'title' in item:
        if get_label(item['venue']) == -1: continue
        year = item['year']
        if year < start_year or year > end_year:
            continue
        time_slot = max((year-start_year)//2, 0)
        num_papers_year[time_slot] += 1

        tokens = word_tokenize(item['title'])
        for token in tokens:
            token_dic[token.lower()] += 1

token_idx = {}
for token in token_dic:
    if token_dic[token] < 5000 and token_dic[token]>200:
        token_idx[token] = len(token_idx)

datapath = '../data/ACM/'
if not os.path.exists(datapath):
    os.makedirs(datapath)
num_class = 4
with open(datapath+'statistics', 'wb') as file:
  pickle.dump((num_task, num_class), file)

def get_feature(text):
    feature = [0 for i in range(len(token_idx))]
    tokens = word_tokenize(text)
    for token in tokens:
        if token.lower() in token_idx:
            feature[token_idx[token.lower()]] += 1
    return feature


all_items = {}
items_year_id = [set() for i in range(num_task)]

for i, (id, item) in enumerate(data.items()):
    if 'venue' in item and 'year' in item and 'title' in item:
        label = get_label(item['venue'])
        if label == -1: continue
        all_items[item['id']] = {'label':label, 'feature': get_feature(item['title']), 'year':year, 'ref': item['ref'] if 'ref' in item else set()}

        year = item['year']
        if year < start_year or year > end_year:
            continue
        time_slot = (year-start_year)//2
        items_year_id[time_slot].add(item['id'])

id2idx = {}
for item_id in all_items:
    id2idx[item_id] = len(id2idx)

g_list = []
for time_slot in range(num_task):
    id2idx_t = {}
    for item_id in items_year_id[time_slot]:
        id2idx_t[item_id] = len(id2idx_t)
    num_new_items = len(id2idx_t)
    for item_id in items_year_id[time_slot]:
        item_year = item['year']
        for rel_id in all_items[item_id]['ref']:
            if rel_id in all_items and all_items[rel_id]['year'] <= item_year:
                if rel_id not in id2idx_t:
                    id2idx_t[rel_id] = len(id2idx_t)

    node_features = [[] for i in range(len(id2idx_t))]
    node_idxs = [-1 for i in range(len(id2idx_t))]
    class_label = [-1 for i in range(len(id2idx_t))]
    g = dgl.DGLGraph()
    g.add_nodes(len(id2idx_t))

    for item_id in id2idx_t:
        idx = id2idx_t[item_id]
#         year_idx_ = all_books[book_id]['year']-start_year
        node_idxs[idx] = id2idx[item_id]
        node_features[idx] = all_items[item_id]['feature']
        class_label[idx] = all_items[item_id]['label']

    for item_id in items_year_id[time_slot]:
        item_year = item['year']
        for rel_id in all_items[item_id]['ref']:
            if rel_id in all_items and all_items[rel_id]['year'] <= item_year:
                g.add_edges(id2idx_t[item_id], id2idx_t[rel_id])
                g.add_edges(id2idx_t[rel_id], id2idx_t[item_id])
    node_features = torch.tensor(node_features)
    class_label = torch.tensor(class_label)
    node_idxs = torch.tensor(node_idxs)
#     node_idxs[idx] = id2idx[book_id]
    g.ndata['num_new_nodes'] = torch.tensor([num_new_items for i in range(len(id2idx_t))])
    g.ndata['x'] = node_features
    g.ndata['y'] = class_label
    g.ndata['node_idxs'] = node_idxs
    print (g)
    g_list.append(g)
    with open(datapath+f'graph_{time_slot}_by_edges', 'wb') as file:
      pickle.dump(g, file)
