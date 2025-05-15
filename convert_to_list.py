import json
import random

data = []
with open("svamp_train_plus.jsonl",'r') as f:
    for line in f:
        one = json.loads(line)
        data.append(one)
# print(data)
#
# def transform_data(items):
#     print(f"previous \n\n{items}")
#     transformed_data = []
#     for key in items.keys():
#         print(key)
#         transformed_data.append({key: items[key]})
#     print(f"after \n\n{transformed_data}")
#     return transformed_data
new_data = []
for item in data:
    item['important'] = [{key: value} for key, value in item['important'].items()]
    new_data.append(item)
# for item in data:


def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

save_to_jsonl(new_data,'svamp_train_plus_add.jsonl')