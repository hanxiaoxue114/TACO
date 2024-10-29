import json
class NumberInt:
    def __init__(self, val):
        self.val = val
    def get_val(self):
        return self.val

null = 'null'

def count_brackets(s):
    l,r = 0,0
    r_idx = -1
    if len(s)>0 and s[0]!='{' and s[0]!='}':
        return l, r, r_idx
    for i, c in enumerate(s):
        if c == '{':
            l += 1
        elif c == "}":
            r += 1
            r_idx = i
    return l, r, r_idx

def process_item(item):
    if 'venue' in item and 'type' in item['venue']:
        item['venue']['type'] = item['venue']['type'].get_val()
    if 'venue' in item and 'raw_zh' in item['venue']:
        del item['venue']['raw_zh']
    if 'year' in item:
        item['year'] = item['year'].get_val()
    if 'n_citation' in item:
        item['n_citation'] = item['n_citation'].get_val()
    if 'pdf' in item:
        del item['pdf']


dataset =  open('../raw-data/dblpv13.json')
dataset_processed = '../raw-data/dblpv13_processed.json'

curr = ""
l_sum, r_sum = 0, 0
item_str = ""
n = 0
count = 0
with open(dataset_processed, "w") as f:
    for x in dataset:
        l, r, r_idx = count_brackets(x)
        l_sum += l
        r_sum += r
        if l_sum != r_sum and l_sum > 0:
            item_str += x
        elif l_sum == r_sum and l_sum > 0:
            item_str += x[:r_idx+1]
            l_sum, r_sum = 0, 0
            item = eval(item_str)
            process_item(item)
            # print (item)
            jsonstr = json.dumps(item)
            f.write(jsonstr + "\n")
            item_str = ""
            n += 1
            if n % 1000 == 0:
                print (n)

# infile = open(dataset_processed)
# for x in infile:
#     print()
#     print ('x:', x)
#     print ('load x:', json.loads(x))
