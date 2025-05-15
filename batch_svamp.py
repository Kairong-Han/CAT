import json
import re

import openai
from openai import ChatCompletion
import concurrent.futures


prompt = """
You need to evaluate the causal importance relationships between tokens in text data from the field of mathematical reasoning. Among them, entities, values, and keywords containing operation symbols are crucial for numerical reasoning. The data is used to train autoregressive models, so tokens that appear later can only see the tokens that come before them. Please output the important tokens for executing mathematical reasoning tasks during training, along with the tokens they should focus on from the preceding context as causal associations (which can be more than one). Present the output JSON string in a dict format,such as {"A":[...],"B":[...],...}. You should only output JSON without other contents. Note that the Answer part is considered important and must be analyzed.

##demo

If they are already at 659 feet and the cave is 762 feet deep. How much farther until they reach the end of the cave?. Answer: 103.0

##output

{
"659 feet":["already"],
"762 feet deep":["the cave"],
"until":["How much farther"],
"Answer":["659 feet","762 feet","until","end of the cave"],
"103.0":["659 feet","and","762 feet","Answer"]
}


##Please out following sentence importance between tokens. The final answer at the end and the corresponding number's importance must always be analyzed (such as 103.0 shown above).

"""
prompt = """
You need to evaluate the causal importance relationships between tokens in text data from the field of mathematical reasoning. Among them, entities, values, and keywords containing operation symbols are crucial for numerical reasoning. The data is used to train autoregressive models, so tokens that appear later can only see the tokens that come before them. Please output the important tokens for executing mathematical reasoning tasks during training, along with the tokens they should focus on from the preceding context as causal associations (which can be more than one). Present the output JSON string in a dict format,such as {"A":[...],"B":[...],...}. You should only output JSON without other contents. Note that the Answer part is considered important and must be analyzed.

##demo

If they are already at 659 feet and the cave is 762 feet deep. How much farther until they reach the end of the cave?. Answer: 103.0

##output

{
"659 feet":["already"],
"762 feet":["the cave"],
"103.0":["659 feet","762 feet","already","until","How much farther","end of the cave"]
}


##Please out following sentence importance between tokens. The final answer at the end and the corresponding number's importance must always be analyzed (such as 103.0 shown above).

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
with open("svamp_train.jsonl","r") as f:
    for line in f:
        data.append(json.loads(line))

pattern = r'(\{.*\})'


skip = 0
with open("svamp_train_plus.jsonl","r") as f:
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
                item['important'] =  {}
        else:
            item['important'] =  {}
        print(item['important'])
        with open('svamp_train_plus.jsonl', 'a+') as f:
                f.write(json.dumps(item) + '\n')
    # break

# def save_to_jsonl(data, filename):
