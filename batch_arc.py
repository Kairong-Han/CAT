import json
import re

import openai
from openai import ChatCompletion
import concurrent.futures


prompt = """
You need to evaluate the causal importance relationships between tokens in text data from the field of reasoning. You only need to consider the tokens that have the greatest impact on the final answer. The data is used to train autoregressive models, so tokens that appear later can only see the tokens that come before them. Please output the important tokens for executing reasoning tasks during training, along with the tokens they should focus on from the preceding context as causal associations (which can be more than one). Present the output JSON string in a dict format,such as {"A":[...],"B":[...],...}. Note that the Answer part is considered important and must be analyzed.
Below I will give you a single-choice question. You need to analyze the most important part of each option for the answer, and together with the answer, form the causal relationship that needs to be considered to generate the answer. Note that only the token behind can notice the previous word, and keep the autoregressive characteristics, such as "option content": "option A/B/C/D". The specific example is as follows:

##demo
Which factor will most likely cause a person to develop a fever?
A. a leg muscle relaxing after exercise
B. a bacterial population in the bloodstream
C. several viral particles on the skin
D. carbohydrates being digested in the stomach
Answer: B

##output
{
"develop a fever":["factor","cause"],
"leg muscle relaxing":["A."],
"bacterial population":["B."],
"viral particles":["C."],
"digested in the stomach":["D."],
"Answer: B":["A.","leg muscle relaxing","B.","bacterial population","C.","viral particles","D.","digested in the stomach"]
}

##Please out following sentence importance between tokens. The final answer at the end and the corresponding number's importance must always be analyzed (such as Answer: B shown above). You should only output JSON string without other contents.

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
            {"role": "user","content": prompt + "\n## Your output:\n"}
        ],
        top_p=0.7,
        temperature=0.9
    )
    return completion.choices[0].message['content']

data = []
import re
with open("arc/arc-E/ARC_E_train.jsonl","r") as f:
    for line in f:
        data.append(json.loads(line))

pattern = r'(\{.*\})'


skip = 0
with open("arc/arc-E/ARC_E_train_add.jsonl","r") as f:
    skip = len(f.readlines())
print(f"skip {skip}")
data = data[skip:]

batch_num = 100
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
        with open('arc/arc-E/ARC_E_train_add.jsonl', 'a+') as f:
                f.write(json.dumps(item) + '\n')

    # break

# def save_to_jsonl(data, filename):
