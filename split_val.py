import json
import os
from sklearn.model_selection import train_test_split

def load_jsonl(filepath):
    """
    Loads a JSONL file into a list of dictionaries.
    
    :param filepath: Path to the JSONL file
    :return: List of loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def split_dataset(dataset, val_ratio=1/4):
    """
    Splits the dataset into a new training set and a validation set based on the given ratio.
    
    :param dataset: List or array-like dataset
    :param val_ratio: Ratio of validation set size to the original dataset size (default: 1/4 for 1:3 split)
    :return: new_train_set, val_set
    """
    new_train_set, val_set = train_test_split(dataset, test_size=val_ratio, random_state=42)
    return new_train_set, val_set

def save_to_jsonl(data, filepath):
    """
    Saves a list of data to a JSONL file.
    
    :param data: List of dictionaries or other JSON-serializable objects
    :param filepath: Path to the output JSONL file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# User-provided directory and filename
directory = "SVAMP"  # Replace with actual directory
filename = "svamp_train.jsonl"  # Replace with actual file name

# Load dataset from file
filepath = os.path.join(directory, filename)
dataset = load_jsonl(filepath)

# Split dataset
if directory == "GSM8k":
    new_train_set, val_set = split_dataset(dataset,1319/7473)
else:
    new_train_set, val_set = split_dataset(dataset)

# Save split datasets back to the directory
save_to_jsonl(new_train_set, os.path.join(directory, "new_train_set.jsonl"))
save_to_jsonl(val_set, os.path.join(directory, "val_set.jsonl"))

print(f"New training set size: {len(new_train_set)}")
print(f"Validation set size: {len(val_set)}")
