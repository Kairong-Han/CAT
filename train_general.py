import math
import random
import sys
from dataloader import CausalAttentionDataset
from datasets import load_dataset,Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import json
import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
import torch
import warnings
import numpy as np
from myloss import compute_attention_constraint_loss_batch,average_attention
import argparse
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters from command line.")
    # 添加命令行参数
    parser.add_argument('--device', type=str, default="cuda:2", help='Device to run the training on (e.g., cuda:0, cpu).')
    parser.add_argument('--log_step', type=int, default=5, help='Number of steps between logging.')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps.')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for input.')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--logout', type=str, default=None, help='Path to the log output file (if specified, all logs will be redirected).')
    parser.add_argument('--mode', type=str, default='base', help='')
    parser.add_argument('--alpha', type=float, default=2.0, help='alpha of the loss calculated.')
    parser.add_argument('--train_dataset', type=str, default="./mawps_train.jsonl", help='train_dataset')
    parser.add_argument('--val_dataset', type=str, default="", help='val_dataset.')
    parser.add_argument('--loss_decay_rate', type=float, default=1, help='')
    parser.add_argument('--loss_type', type=str, default="div", help='')
    parser.add_argument('--taskname', type=str, default="", help='task name')
    parser.add_argument('--bit', type=str, default="16bit", help='')
    parser.add_argument('--Lora', type=str, default="False", help='use lora')
    parser.add_argument('--model', type=str, default="", help='model name')
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
    if args.val_dataset == "none":
        args.val_dataset=""
    if args.logout:
        redirect_logs(args.logout)
    print(json.dumps(vars(args), indent=4))
    print("\n>>>> Log redirection successful!" if args.logout else "\n>>>> Log redirection not specified.")


val_data = []
if args.val_dataset != "":
    with open(args.val_dataset,'r') as f:
        for line in f:
            val_data.append(json.loads(line))


def Causal_collate_fn(batch):
    texts,_,masks,labels = zip(*batch)
    max_text_len = max([text.size(0) for text in texts])+1
    padded_texts = torch.stack([torch.cat([text, torch.full((max_text_len - text.size(0),), tokenizer.pad_token_id)]) for text in texts])
    padded_labels = torch.stack([torch.cat([label, torch.full((max_text_len - label.size(0),), -100)]) for label in labels])
    padded_masks = []
    for mask in masks:
        padded_mask = torch.zeros((max_text_len, max_text_len), dtype=mask.dtype)
        padded_mask[:mask.size(0), :mask.size(1)] = mask
        padded_masks.append(padded_mask)
    padded_masks = torch.stack(padded_masks)
    return padded_texts, padded_masks, padded_labels



model_path_dict = {
    # model paths
}

model_name = model_path_dict[args.model]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'right'

if args.bit == '8bit':
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
elif args.bit == '16bit':
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)

if args.Lora == "True":
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1,target_modules=["gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj"]
    )
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())


device = args.device
log_step = args.log_step
accumulation_steps = args.accumulation_steps
batchsize = args.batchsize
lr = args.lr
weight_decay = args.weight_decay
max_length = args.max_length
num_epoch = args.num_epoch
alpha = args.alpha
mode = args.mode
loss_decay_rate = args.loss_decay_rate
loss_type = args.loss_type
model.to(device)


if args.model in ["Llama-3.1-8B-Instruct","Llama-3.2-1B-Instruct"]:
    dataload = CausalAttentionDataset(args.train_dataset,max_length=max_length,tokenizer=tokenizer,question='',label='',Llama3_flag=True)
elif args.model in ["TinyLlama-1.1B"]:
    dataload = CausalAttentionDataset(args.train_dataset,max_length=max_length,tokenizer=tokenizer,question='',label='',Llama_flag=True)
else:
    dataload = CausalAttentionDataset(args.train_dataset,max_length=max_length,tokenizer=tokenizer,question='',label='')

train_dataloader = torch.utils.data.DataLoader(dataload, batch_size=batchsize, shuffle=True,collate_fn=Causal_collate_fn)



loss_values = []
acc_values = []

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
num_training_steps = num_epoch * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps/accumulation_steps)
lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=int(num_training_steps/accumulation_steps))
progress_bar = tqdm.tqdm(range(int(num_training_steps)))
step = 0
max_acc = 0
final_acc = 0
min_epoch_loss = 1000000000
val_max_acc = 0
for epoch in range(num_epoch):
    optimizer.zero_grad()
    epoch_loss = 0
    model.train()
    for batch in train_dataloader:
        prompt = batch[0] 
        labels = batch[2]
        outputs = model(prompt.to(device), labels=labels.to(device),output_attentions=True)
        loss_total = outputs.loss
        new_attention = average_attention(outputs.attentions)
        loss2 = compute_attention_constraint_loss_batch(batch[1][:,:,:],new_attention[:,:,:],alpha,Amp=math.exp(-loss_decay_rate * epoch),print_detail=False,loss_type=args.loss_type)
        loss_total += loss2
        epoch_loss += loss_total.item()
        loss_total.backward()
        progress_bar.update(1)
        step+=1
        if step % accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if step%log_step == 0:
            print('\n epoch {}, step {}, loss {} lr {}'.format(epoch, progress_bar.n, loss_total.item(),lr_scheduler.get_last_lr()[0]))# lr_scheduler.get_last_lr()[0]
    if args.val_dataset=="" and epoch_loss<min_epoch_loss :
        print(f"before min loss {min_epoch_loss}, after min loss {epoch_loss}")
        min_epoch_loss = epoch_loss
        if epoch >= num_epoch/2-1:
            save_path="path/to/save"
            model.save_pretrained(save_path)
            print(f"Model saved in {save_path}")
    elif args.val_dataset !="" :

        def get_acc():
            # Calculate the validation set accuracy
            return
        save_path="path/to/save"
        acc1 = get_acc()
        final_acc = acc1
        if val_max_acc < acc1:
            val_max_acc = acc1
            model.save_pretrained(save_path)
            print(f"Model saved in {save_path} with accuracy: {acc1}")

print(f"final val acc = {final_acc}")
print(f"max val acc = {val_max_acc}")
