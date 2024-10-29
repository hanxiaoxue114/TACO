import json

def read_line(line):
    if line[:2] == '#*':
        return {'title':line[2:-1]}
    if line[:2] == "#@":
        return {'authors':line[2:-1]}
    if line[:2] == "#c":
        return {'venue':line[2:-1]}
    if line[:2] == '#t':
        return {'year':int(line[2:-1])}
    if line[:6] == '#index':
        return {'id': line[6:-1]}
    if line[:2] == '#%':
        return {'ref':set([line[2:-1]])}
    return None

all_items = {}
curr_item = {}
for i, x in enumerate(open('../raw-data/acm.txt')):
    if x == '\n':
        if 'id' in curr_item:
            all_items[curr_item['id']] = curr_item
        curr_item = {}
    else:
        content = read_line(x)
        if content == None:
            continue
        if 'ref' in content and 'ref' in curr_item:
            curr_item['ref'].update(content['ref'])
        else:
            curr_item.update(content)

    if i%10000==0:
        print (i)

import pickle
with open('../raw-data/ACM_processed.pickle', 'wb') as handle:
    pickle.dump(all_items, handle)
