import json
import re

import openai
from openai import ChatCompletion
import concurrent.futures



prompt = """
You need to evaluate the causal importance relationships between tokens in text data from the field of mathematical reasoning. Among them, entities, values, and keywords containing operation symbols are crucial for numerical reasoning. The data is used to train autoregressive models, so tokens that appear later can only see the tokens that come before them. Please output the important tokens for executing mathematical reasoning tasks during training, along with the tokens they should focus on from the preceding context as causal associations (which can be more than one). Present the output JSON string in a dict format,such as {"A":[...],"B":[...],...}. You should only output JSON without other contents. Note that the Answer part is considered important and must be analyzed.

##demo

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer: Natalia sold 48\/2 = <<48\/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72


##output

{
"in April":["48"],
"in May":["half as many clips","48\/2 = <<48\/2=24>>24 clips","48"],
"72 clips":["How many clips","sell altogether","48+24","in April","in May"],
"#### 72":["How many clips","in April and May","48+24","72 clips"]
}


##Please out following sentence importance between tokens. The final answer at the end and the corresponding number's importance must always be analyzed (such as #### 72 shown above).Please try to use the most refined causal characteristics to summarize the causal process of the answer

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
            {"role": "user","content": prompt+"\n\n###Your Output:"}
        ],
        top_p=0.7,
        temperature=0.9
    )
    return completion.choices[0].message['content']

data = []
import re
with open("math-evaluation-harness-main/data/gsm8k/train.jsonl","r") as f:
    for line in f:
        item = json.loads(line)
        item['input'] = item['question']+" Answer: "
        item['target'] = item['answer']
        data.append(item)

pattern = r'(\{.*\})'

skip = 0
with open("gsm8k_train_plus.jsonl","r") as f:
    skip = len(f.readlines())
print(f"skip {skip}")
data = data[skip:]

batch_num = 500
for idx in range(0,len(data),batch_num):
    end = idx+batch_num
    if end >= len(data):
        end = len(data)
    # item['input'] = item["question"]
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
        with open('gsm8k_train_plus.jsonl', 'a+') as f:
                f.write(json.dumps(item) + '\n')

    # break

# def save_to_jsonl(data, filename):
