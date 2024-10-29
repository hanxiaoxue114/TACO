import os
import json
import gzip
from urllib.request import urlopen
from nltk.tokenize import word_tokenize
from collections import defaultdict

def get_feature(text):
    feature = [0 for i in range(len(token_idx))]
    tokens = word_tokenize(text)
    for token in tokens:
        if token.lower() in token_idx:
            feature[token_idx[token.lower()]] += 1
    return feature

datafile = '../raw-data/meta_Kindle_Store.json.gz'

data = []
with gzip.open(datafile) as f:
    for l in f:
        data.append(json.loads(l.strip()))

# total length of list, this number equals total number of products
print(len(data))
# first row of the list
print(data[0])

token_dic = defaultdict(int)
for book in data:
    if book['title'] != '':
        tokens = word_tokenize(book['title'])
        for token in tokens:
            token_dic[token.lower()] += 1
token_idx = {}
for token in token_dic:
    if token_dic[token] < 30000 and token_dic[token]>1000:
        token_idx[token] = len(token_idx)


category_dic = defaultdict(int)
for book in data:
    if book['title'] !=  '' and book['category'] != []:
        for category in book['category']:
            category_dic[category] += 1

category_idx = {}
for category in category_dic:
    if category_dic[category]>15000 and category_dic[category]<100000:
        category_idx[category] = len(category_idx)

target_years = [2012, 2013, 2014, 2015, 2016]
start_year = target_years[0]
book_category_years = [[0 for _ in category_idx] for y in target_years]
for book in data:
    if book['title'] !=  '':
        if 'Publication Date:' in book['details'] and book['details']['Publication Date:']:
            year = int(book['details']['Publication Date:'].split(', ')[-1])
            if year not in target_years: continue
            year_idx = year - start_year
            for category in book['category']:
                if category in category_idx:
                    book_category_years[year_idx][category_idx[category]] += 1
                    break

import pickle
datapath = '../data/kindle/'
if not os.path.exists(datapath):
    os.makedirs(datapath)
num_task = len(target_years)
num_class = len(category_idx)
with open(datapath+'statistics', 'wb') as file:
  pickle.dump((num_task, num_class), file)


all_books = {}
book_year_id = [set() for i in range(6)]
for book in data:
    if book['title'] !=  '':
        if 'Publication Date:' in book['details'] and book['details']['Publication Date:']:
            year = int(book['details']['Publication Date:'].split(', ')[-1])
#             if year not in target_years: continue
            year_idx = max(year - start_year, 0)
            if year_idx >= num_task: continue
            for category in book['category']:
                if category in category_idx:
                    feature = get_feature(book['title'])
                    label = category_idx[category]
                    related_books = set(book['also_view']) | set(book['also_buy'])
                    all_books[book['asin']] = {'year':year, 'feature':feature, 'label':label, 'related_books':related_books}
                    book_year_id[year_idx].add(book['asin'])
                    break


id2idx = {}
for book_id in all_books:
    id2idx[book_id] = len(id2idx)


g_list = []
for year_idx in range(6):
    id2idx_t = {}
    for book_id in book_year_id[year_idx]:
        id2idx_t[book_id] = len(id2idx_t)
    num_new_books = len(id2idx_t)
    for book_id in book_year_id[year_idx]:
        for rel_id in all_books[book_id]['related_books']:
            if rel_id in all_books and all_books[rel_id]['year'] <= year_idx+start_year:
                if rel_id not in id2idx_t:
                    id2idx_t[rel_id] = len(id2idx_t)

    node_features = [[] for i in range(len(id2idx_t))]
    node_idxs = [-1 for i in range(len(id2idx_t))]
    class_label = [-1 for i in range(len(id2idx_t))]
    g = dgl.DGLGraph()
    g.add_nodes(len(id2idx_t))
    for book_id in id2idx_t:
        idx = id2idx_t[book_id]
        node_idxs[idx] = id2idx[book_id]
        node_features[idx] = all_books[book_id]['feature']
        class_label[idx] = all_books[book_id]['label']

    for book_id in book_year_id[year_idx]:
        for rel_id in all_books[book_id]['related_books']:
            if rel_id in all_books and all_books[rel_id]['year'] <= year_idx+start_year:
                g.add_edges(id2idx_t[book_id], id2idx_t[rel_id])
                g.add_edges(id2idx_t[rel_id], id2idx_t[book_id])
    node_features = torch.tensor(node_features)
    class_label = torch.tensor(class_label)
    node_idxs = torch.tensor(node_idxs)
    g.ndata['num_new_nodes'] = torch.tensor([num_new_books for i in range(len(id2idx_t))])
    node_mask = torch.tensor([False for i in range(len(id2idx_t))])
    node_mask[torch.tensor([i for i in range(num_new_books)])] = True
    g.ndata['new_node_mask'] = node_mask
    g.ndata['x'] = node_features
    g.ndata['y'] = class_label
    g.ndata['node_idxs'] = node_idxs
    print (g)
    g_list.append(g)
    with open(datapath+f'graph_{year_idx}_by_edges', 'wb') as file:
      pickle.dump(g, file)


node_features = [[] for i in range(len(id2idx))]
class_label = [-1 for i in range(len(id2idx))]
g = dgl.DGLGraph()
g.add_nodes(len(id2idx))
for time_slot in range(6):
    for book_id in book_year_id[time_slot]:
        book_year = all_books[book_id]['year']
        year_idx = book_year-start_year
        idx = id2idx[book_id]
        node_features[idx] = all_books[book_id]['feature']
        class_label[idx] = all_books[book_id]['label']
        for rel_id in all_books[book_id]['related_books']:
            if rel_id in id2idx and all_books[rel_id]['year'] <= year_idx+start_year:
                g.add_edges(idx, id2idx[rel_id])
                g.add_edges(id2idx[rel_id], idx)
node_features = torch.tensor(node_features)
class_label = torch.tensor(class_label)
g.ndata['x'] = node_features
g.ndata['y'] = class_label
# with open(datapath+'full_graph', 'wb') as file:
#   pickle.dump(g, file)
