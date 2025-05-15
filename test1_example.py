import random
import sys
from dataloader import CausalAttentionDataset
from datasets import load_dataset,Dataset
from peft import PeftModelForCausalLM,LoraConfig,PeftModel,TaskType,get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,Trainer, TrainingArguments,DataCollatorForLanguageModeling
import json
import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import re
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import argparse
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters from command line.")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run the training on (e.g., cuda:0, cpu).')
    parser.add_argument('--log_step', type=int, default=5, help='Number of steps between logging.')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps.')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for input.')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--logout', type=str, default=None, help='Path to the log output file (if specified, all logs will be redirected).')
    parser.add_argument('--mode', type=str, default='base', help='base means baseline. our means our method.')
    parser.add_argument('--alpha', type=float, default=2.0, help='alpha of the loss calculated.')
    parser.add_argument('--train_dataset', type=str, default="./mawps_train.jsonl", help='train_dataset')
    parser.add_argument('--test_dataset', type=str, default="./mawps_test.jsonl", help='test_dataset.')
    parser.add_argument('--val_dataset', type=str, default="", help='val_dataset.')
    parser.add_argument('--loss_decay_rate', type=float, default=1, help='test_dataset.')
    parser.add_argument('--loss_type', type=str, default="div", help='test_dataset.')
    parser.add_argument('--taskname', type=str, default="", help='task name.')
    parser.add_argument('--model', type=str, default="", help='task name.')
    parser.add_argument('--origin_model', type=str, default="", help='task name.')
    parser.add_argument('--save_wrong', type=str, default="No", help='task name.')
    parser.add_argument('--bit', type=str, default="16bit", help='16bit default')
    parser.add_argument('--Lora', type=str, default="False", help='use lora?')
    parser.add_argument('--delete_after_test', type=str, default="False", help='')
    args = parser.parse_args()
    return args

def redirect_logs(log_file):
    """
    Redirect stdout and stderr to a log file.
    """
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout

if __name__ == "__main__":

    args = parse_args()

    if args.logout:
        redirect_logs(args.logout)

    print(json.dumps(vars(args), indent=4))
    print("\n>>>> Log redirection successful!" if args.logout else "\n>>>> Log redirection not specified.")


test_data = []
with open(args.test_dataset,'r') as f:
    for line in f:
        test_data.append(json.loads(line))
model_path_dict = {
    "Llama-3.1-8B-Instruct":"",
    "TinyLlama-1.1B":"",
    "Qwen-2.5-1.5B":"",
}

import json

def save_list_to_jsonl(jsonl_filename, data_list):

    with open(jsonl_filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def my_general_test(model, tokenizer, test_dataset,max_length,batch_size=3,check_prediction=None):
    correct = 0
    total = len(test_dataset)
    print(f" {len(test_dataset)}")
    model.eval()
    batch_size = 64

    for i in tqdm.trange(0, total, batch_size):

        batch_samples = test_dataset[i:i + batch_size]

        input_texts = [sample['input'] for sample in batch_samples]
        label_texts = [sample['target'] for sample in batch_samples]

        inputs = tokenizer(input_texts, return_tensors='pt', padding=True,padding_side='left', truncation=True, max_length=max_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, num_return_sequences=1, do_sample=False,
                                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for prediction, input_text, label_text in zip(predictions, input_texts, label_texts):
            if check_prediction(prediction, input_text, label_text):
                correct += 1
    accuracy = correct / total
    print(f" {accuracy:.2%} = {correct}/{total}")
    model.train()
    return accuracy,0


model_name = model_path_dict[args.origin_model]
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'  

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

def parse_number(path):
    match = re.search(r'\b(\d+(?:e[-+]?\d+)?)\b', path)
    if match:
        num_str = match.group(1)
        return float(num_str)
    return None


def check(prediction, input_text, label_text):

    import re
    output_text = prediction.replace(input_text, "").strip()
    if output_text == label_text:
        return True
    else:
        print(f"output_text : {output_text} <<>> label_text : {label_text}")
        return False
    match = re.search(r'#### (.*)', output_text)
    # match2 = re.search(r'#### (.*)', label_text)
    if match:

        if match.group(1) == label_text:
            return match.group(1) == label_text
        else:
            print(f"output_text : {match.group(1)} <<>> label_text : {label_text}")
            return False
    else:
        print(f"output_text : {output_text} <<>> label_text : {label_text}")
        return False
    if output_text == label_text:
        return True
    else:
        print(f"output_text : {output_text} <<>> label_text : {label_text}")
        return False
    import re

    match = re.search(r'Answer: (\d+)$', prediction)
    if match:

        extracted_number = float(match.group(1))

        return extracted_number == true_number
    return False

def check_gsm(prediction, input_text, label_text):

    import re
    output_text = prediction.replace(input_text, "").strip()
    match = re.search(r'#### (.*)', output_text)
    if match:
        if match.group(1).strip() == label_text.strip():
            return True
        else:
            print(f"output_text : {match.group(1).strip() } <<>> label_text : {label_text.strip() }")
            return False
    else:
        print(f"output_text : {output_text} <<>> label_text : {label_text}")
        return False
    if output_text == label_text:
        return True
    else:
        print(f"output_text : {output_text} <<>> label_text : {label_text}")
        return False
    import re

    match = re.search(r'Answer: (\d+)$', prediction)
    if match:

        extracted_number = float(match.group(1))

        return extracted_number == true_number
    return False


model_name_read = args.model
print(f'load from {model_name_read}')

if args.Lora == "True":
    if args.origin_model in ["Llama-3.1-8B-Instruct"] and args.taskname=="GSM8K" :
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_read,torch_dtype=torch.float16)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_read)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name_read)


device = args.device
log_step = args.log_step
model.to(device)
max_length = args.max_length
batchsize = args.batchsize

if args.taskname.lower().strip() == "gsm8k":
    acc,_ = my_general_test(model,tokenizer,test_data,max_length,batchsize,check_gsm)
else:
    acc,_ = my_general_test(model,tokenizer,test_data,max_length,batchsize,check)
    if args.delete_after_test == "True":
        import shutil
        folder_path = model_name_read
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted the model: {folder_path}")
        except FileNotFoundError:
            print("model not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
print(f"test acc: {acc}")


