import json
import os.path
import re

import openai
from openai import ChatCompletion
import concurrent.futures


prompt = """
You need to evaluate the causal importance relationships between tokens in text data from the field of mathematical reasoning. Among them, entities, values, and keywords containing operation symbols are crucial for numerical reasoning. The data is used to train autoregressive models, so tokens that appear later can only see the tokens that come before them. Please output the important tokens for executing mathematical reasoning tasks during training, along with the tokens they should focus on from the preceding context as causal associations (which can be more than one). Present the output JSON string in a dict format,such as {"A":[...],"B":[...],...}. You should only output JSON without other contents. Note that the Answer part is considered important and must be analyzed.

##demo

Penny's class is going to Animaland, the largest zoo on earth, for their science field trip. The zoo has a variety of wild animals in captivity. Their first destination was the aviary. The aviary has 3 species of eagles on show that day. They have 20 Bald Eagles, 15 Harpy Eagles, and 30 Crowned Eagles. How many eagles are on display that day? Answer: 65 (eagles)

##output

{
"3 species of eagles":["The aviary has"],
"Bald Eagles":["20",3 species],
"Harpy Eagles":["15",3 species],
"Crowned Eagles":["30",3 species],
"65 (eagles)":["20 Bald Eagles","15 Harpy Eagles","30 Crowned Eagles","How many eagles"],
}


##Please out following sentence importance between tokens. The final answer at the end and the corresponding number's importance must always be analyzed (such as 65 shown above).

"""

def batch_process(prompts):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(Chat, prompts))
    return results

def Chat(prompt):
    completion = ChatCompletion.create(
        model="GLM-4-air",
        messages=[
            {"role": "system", "content": "You are an assistant who helps me discover the causal relationships between tokens in the training data of the autoregressive model."},
            {"role": "user","content": prompt}
        ],
        top_p=0.7,
        temperature=0.9
    )
    return completion.choices[0].message['content']

data = []
with open("asdiv_train.jsonl","r") as f:
    for line in f:
        data.append(json.loads(line))

pattern = r'(\{.*\})'


skip = 0
if os.path.exists("asdiv_train_plus.jsonl"):
    with open("asdiv_train_plus.jsonl","r") as f:
        skip = len(f.readlines())
print(f"skip {skip}")
data = data[skip:]

batch_num = 500
for idx in range(0,len(data),batch_num):
    end = idx+batch_num
    if end >= len(data):
        end = len(data)
    prompts = [prompt + item["input"]+" "+item["target"] for item in data[idx:end]]
    responses = batch_process(prompts)
    # resp = Chat(prompt + item["input"]+" "+item["target"])
    # print(resp)
    for resp in responses:
        item = data[idx+responses.index(resp)]
        matches = re.findall(pattern, resp,re.DOTALL)
        if matches:
            try:
                item['important']=json.loads(matches[0])
            except Exception as e:
                print(f"{e}")
                item['important'] = {}
        else:
            item['important'] = {}
        print(item['important'])
        with open('asdiv_train_plus.jsonl', 'a+') as f:
                f.write(json.dumps(item) + '\n')

    # break

# def save_to_jsonl(data, filename):
