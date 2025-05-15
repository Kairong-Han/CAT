
from torch.utils.data import Dataset
import json
# from transformers import AutoModelForCausalLM, Trainer, TrainingArguments,DataCollatorForLanguageModeling
import json
import torch
# import warnings
import numpy as np



class CausalAttentionDataset(Dataset):
    def __init__(self, filename , max_length: int, tokenizer,question='question',label = 'label',mask = 'important',Llama_flag=False,Answer_idx=False,Only_output=False,Llama3_flag=False,Qwen3_flag=False):
        self.data_list = []
        with open(filename,'r') as f:
            for line in f:
                self.data_list.append(json.loads(line))
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.question = question
        self.label = label
        self.mask = mask
        self.Llama_flag = Llama_flag
        self.Llama3_flag = Llama3_flag
        self.Qwen3_flag = Qwen3_flag
        self.Answer_idx = Answer_idx
        self.Only_output = Only_output

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, id):
        # first find the idx of answer, the first number in the self.cum_answer is larger than the idx.
        question = self.data_list[id][self.question]
        answer = self.data_list[id][self.label]
        # text = f"{question}{answer}"
        if self.Qwen3_flag:
            text = f"{question}{answer}{self.tokenizer.pad_token}{self.tokenizer.pad_token}{self.tokenizer.pad_token}"
        else:
            text = f"{question}{answer}{self.tokenizer.pad_token}"
        encoding = self.tokenizer(text,truncation=True,max_length=self.max_length,return_tensors="pt")
        if self.Llama_flag:
            mask = self.LLamaText_to_Mask(text,self.data_list[id][self.mask])
        elif self.Llama3_flag:
            mask = self.LLama3Text_to_Mask(text,self.data_list[id][self.mask])
        else:
            mask = self.Text_to_Mask(text,self.data_list[id][self.mask])
        labels=self.tokenizer(text,truncation=True,max_length=self.max_length,return_tensors="pt")["input_ids"][0]
        len_question=len(self.tokenizer(question,truncation=True,max_length=self.max_length,return_tensors="pt")['input_ids'][0])
        labels[:len_question-1] = -100
        return encoding.input_ids.squeeze(), torch.tensor(0), torch.tensor(mask), labels

    def Text_to_Mask(self,input_text,adj_list):
       
        def find_sublist(lst, sublist):
            lst_np = np.array(lst)
            sublist_np = np.array(sublist)
            ret = []
            
            for i in range(len(lst) - len(sublist) + 1):
                if np.array_equal(lst_np[i:i + len(sublist)], sublist_np):
                    ret.append(i)

            return ret  
        tokens = self.tokenizer(input_text,truncation=True,max_length=self.max_length,return_tensors='pt')

        length = len(tokens['input_ids'][-1])
        mask = np.zeros((length, length))
        try:

            for item in adj_list:
                for key, values in item.items():

                    key_tokens = self.tokenizer(" "+key, return_tensors='pt')["input_ids"][-1]
                    key_ids = find_sublist(tokens['input_ids'][-1], key_tokens)
                    if input_text.startswith(key):
                        key_tokens = self.tokenizer(key, return_tensors='pt')["input_ids"][-1]
                        key_ids = key_ids  + find_sublist(tokens['input_ids'][-1], key_tokens)

                    if len(key_ids)==0:
                        continue

                    for value in values:

                        value_tokens = self.tokenizer(" "+value, return_tensors='pt')["input_ids"][-1]
                        v_ids = find_sublist(tokens['input_ids'][-1], value_tokens)
                        if input_text.startswith(value):
                            value_tokens = self.tokenizer(value, return_tensors='pt')["input_ids"][-1]
                            v_ids = v_ids + find_sublist(tokens['input_ids'][-1], value_tokens)
                        
                        if len(v_ids)==0:
                            continue

                        for key_pos in key_ids:
                            for value_pos in v_ids:
 
                                if key_pos > value_pos and key_pos> value_pos+len(value_tokens):
                                    mask[key_pos-1:key_pos+len(key_tokens)-1, value_pos:value_pos+len(value_tokens)] = 1
            return mask
        except Exception as e:
            print(f"exception {e}")
            return np.zeros((length, length))
        
    def LLamaText_to_Mask(self,input_text,adj_list):

        def find_sublist(lst, sublist):
            lst_np = np.array(lst)
            sublist_np = np.array(sublist)
            ret = []

            for i in range(len(lst) - len(sublist) + 1):
                if np.array_equal(lst_np[i:i + len(sublist)], sublist_np):
                    ret.append(i)

            return ret 
        tokens = self.tokenizer(input_text, truncation=True, max_length=self.max_length, return_tensors='pt')

        length = len(tokens['input_ids'][-1])
        mask = np.zeros((length, length))
        triangular = torch.tril(torch.ones(length,length))
        try:

            for item in adj_list:
                for key, values in item.items():
                    key_tokens = self.tokenizer(key, return_tensors='pt')["input_ids"][-1][1:]
                    key_ids = find_sublist(tokens['input_ids'][-1], key_tokens)
                    if len(key_ids)==0:
                        continue
                    for value in values:
                        value_tokens = self.tokenizer(value, return_tensors='pt')["input_ids"][-1][1:]
                        v_ids = find_sublist(tokens['input_ids'][-1], value_tokens)
                        if len(v_ids)==0:
                            continue
                        for key_pos in key_ids:
                            for value_pos in v_ids:
                                if key_pos > value_pos and key_pos> value_pos+len(value_tokens):
                                    mask[key_pos-1:key_pos+len(key_tokens)-1, value_pos:value_pos+len(value_tokens)] = 1
            mask[triangular==0]=0
            return mask
        except Exception as e:
            print(f"exception {e}")
            return np.zeros((length, length))

    def LLama3Text_to_Mask(self,input_text,adj_list):

        def find_sublist(lst, sublist):
            lst_np = np.array(lst)
            sublist_np = np.array(sublist)
            ret = []

            for i in range(len(lst) - len(sublist) + 1):
                if np.array_equal(lst_np[i:i + len(sublist)], sublist_np):
                    ret.append(i)
                    # return i 
            return ret  
        tokens = self.tokenizer(input_text, truncation=True, max_length=self.max_length, return_tensors='pt')

        length = len(tokens['input_ids'][-1])
        mask = np.zeros((length, length))
        triangular = torch.tril(torch.ones(length,length))
        try:

            for item in adj_list:
                for key, values in item.items():

                    key_tokens = self.tokenizer(" "+key, return_tensors='pt')["input_ids"][-1][1:]
                    key_ids = find_sublist(tokens['input_ids'][-1], key_tokens)
                    if  input_text.startswith(key):
                        key_tokens = self.tokenizer(key, return_tensors='pt')["input_ids"][-1][1:]
                        key_ids = key_ids + find_sublist(tokens['input_ids'][-1], key_tokens)
                    
                    if len(key_ids)==0:
                        continue

                    for value in values:
                        value_tokens = self.tokenizer(" "+value, return_tensors='pt')["input_ids"][-1][1:]
                        v_ids = find_sublist(tokens['input_ids'][-1], value_tokens)
                        if input_text.startswith(value):
                            value_tokens = self.tokenizer(value, return_tensors='pt')["input_ids"][-1][1:]
                            v_ids = v_ids + find_sublist(tokens['input_ids'][-1], value_tokens)                            

                        if len(v_ids)==0:
                            continue

                        for key_pos in key_ids:
                            for value_pos in v_ids:

                                if key_pos > value_pos and key_pos> value_pos+len(value_tokens):
                                    mask[key_pos-1:key_pos+len(key_tokens)-1, value_pos:value_pos+len(value_tokens)] = 1
            mask[triangular==0]=0
            return mask
        except Exception as e:
            print(f"exception {e}")
            return np.zeros((length, length))
          
def is_lower_triangular(matrix):

    matrix = np.array(matrix)

    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(i + 1, cols):
            if matrix[i, j] != 0:
                return False
    return True