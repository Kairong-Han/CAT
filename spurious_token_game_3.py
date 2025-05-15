import json
import random
from collections import Counter
random.seed(42)

question_template = "Here is the statistical data for a person. Please predict the probability of cancer. "

def generate_values(causal_words, spurious_words, unrelated_words,spurious_mapping):
    """
    Generate random values for causal words and set the values of spurious and unrelated words to 0.

    Args:
        causal_words (list): List of causal words.
        spurious_words (list): List of spurious words.
        unrelated_words (list): List of unrelated words.

    Returns:
        dict: A dictionary mapping each word to its corresponding value.
    """
    values = {}

    ratio = {"Yellow fingers":1.5, "Clothing size":1, "Hormones":0.5}
    for word in causal_words:
        values[word] = random.randint(1, 10)  # Assign a random value to causal words.
        values[spurious_mapping[word]] = int(values[word]*ratio[spurious_mapping[word]])
        print(values[word],values[spurious_mapping[word]])
    for word in unrelated_words:
        values[word] = random.randint(1, 10)  # Spurious and unrelated words have a value of 0.
    return values

def calculate_label(causal_words, values):
    """
    Calculate the label based on the counts of causal words and their assigned values.
    Args:
        word_counts (dict): A dictionary of word counts.
        values (dict): A dictionary mapping words to their values.

    Returns:
        float: The calculated label.
        ["Smoking", "Weight", "Exercise"]
    """
    value = 1.2*values["Smoking"]+0.7*values["Weight"]-values["Exercise"]
    if value >  7.2:
        label = "High Risk"
    elif 2.7<= value <=2.7:
        label = "Middle Risk"
    else:
        label = "Low Risk"
    return label

def generate_data(num_samples, causal_words, spurious_mapping, unrelated_words, ratio):
    """
    Generate a dataset of random strings and their corresponding labels.

    Args:
        num_samples (int): Number of strings to generate.
        causal_words (list): List of causal words.
        spurious_mapping (dict): Mapping of causal words to their spurious counterparts.
        unrelated_words (list): List of unrelated words.
        ratio (float): Ratio of causal words to their spurious counterparts.
    Returns:
        list: A list of tuples where each tuple contains a generated string and its label.
    """
    dataset = []

    for _ in range(num_samples):
        # Generate a random number of each causal word.
        values = generate_values(causal_words, list(spurious_mapping.values()), unrelated_words, spurious_mapping)
        all_words = causal_words + list(spurious_mapping.values()) + unrelated_words
        random.shuffle(all_words)
        # values = {i:random.randint(1, 10) for i in all_words}
        all_values = []
        for item in all_words:
            all_values.append(f"{item}: {values[item]}")
        generated_string = ", ".join(all_values)
        # Calculate the label.
        label = calculate_label(causal_words, values)
        dataset.append((question_template+generated_string, label))

    dataset2 = []
    #generate ood test set
    for _ in range(400):
        # values = generate_values(causal_words, list(spurious_mapping.values()), unrelated_words, spurious_mapping)
        all_words = causal_words + list(spurious_mapping.values()) + unrelated_words
        values = {i : random.randint(1,10) for i in all_words}
        random.shuffle(all_words)
        all_values = []
        for item in all_words:
            all_values.append(f"{item}: {values[item]}")
        generated_string = ", ".join(all_values)
        # Calculate the label.
        label = calculate_label(causal_words, values)
        dataset2.append((question_template+generated_string, label))

    return dataset,dataset2

causal_words = ["Smoking", "Weight", "Exercise"]
spurious_words = ["Yellow fingers", "Clothing size", "Hormones"]
unrelated_words = ["Certain gene", "Room size"]


spurious_mapping = dict(zip(causal_words, spurious_words))
print(f"spurious_mapping {spurious_mapping}")

# Generate the dataset
dataset,ood_dataset = generate_data(
    num_samples=2000,  # Number of samples to generate.
    causal_words=causal_words,
    spurious_mapping=spurious_mapping,
    unrelated_words=unrelated_words,
    ratio=0.5  # Ratio of causal to spurious words.
)
train = dataset[400:]
test = dataset[:400]

output_file = "spurious_token_game_train3.jsonl"


import re
pattern = r"(Smoking|Weight|Exercise):\s*(\d+)"

with open(output_file, 'w') as f:
    for generated_string, label in train:

        match = re.findall(pattern, generated_string)
        print(match)
        for item in match:
            print(f"{item[0]}: {item[1]}")
       
        result = {
            "input": f"{generated_string} Answer: ",
            "target": str(label),
            "important":[{f'{str(label)}':[f"{item[0]}: {item[1]}" for item in match]}]
        }
        f.write(json.dumps(result) + "\n")

output_file = "spurious_token_game_test3.jsonl"


with open(output_file, 'w') as f:
    for generated_string, label in test:

        result = {
            "input": f"{generated_string} Answer: ",
            "target": str(label)
        }

        f.write(json.dumps(result) + "\n")

with open('spurious_token_game_ood3.jsonl', 'w') as f:
    for generated_string, label in ood_dataset:

        result = {
            "input": f"{generated_string} Answer: ",
            "target": str(label)
        }
        f.write(json.dumps(result) + "\n")
print(f"Dataset saved to {output_file}")


# Print the generated dataset
# for i, (string, label) in enumerate(dataset):
#     print(f"Sample {i + 1}:\nString: {string}\nLabel: {label}\n")
