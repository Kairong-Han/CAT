# <p style="display: flex; align-items: center; gap: 0.5em;"><img src="cat.svg" alt="CAT Logo" height="24"/>CAT: Causal Attention Tuning For Injecting Fine-grained Causal Knowledge into Large Language Models</p>

## Description

Large Language Models (LLMs) have achieved remarkable success across various domains. However, a fundamental question remains: Can LLMs effectively utilize causal knowledge for prediction and generation? Through empirical studies, we find that LLMs trained directly on large-scale data often capture spurious correlations rather than true causal relationships, leading to suboptimal performance, especially in out-of-distribution (OOD) scenarios. 
To address this challenge, we propose Causal Attention Tuning (CAT), a novel approach that injects fine-grained causal knowledge into the attention mechanism.
We propose an automated pipeline that leverages human priors and domain causal graphs to automatically generate token-level causal signals and introduce a Re-Attention mechanism to guide training, helping the model focus on causal structures while mitigating noise and biases in attention scores.
Experimental results on our proposed STG benchmark and multiple downstream tasks demonstrate that our approach effectively leverages causal knowledge for prediction and remains robust in OOD scenarios, proving its effectiveness.

![Alt text](image-1.png)

## Directory Structure

dataset/

├── ARC_E/        

├── ASDiv/      

├── GSM8k/       

├── MAWPS/         

├── STG/              # (our stg dataset)

├── STG_H/            # (our stg—_H dataset)

├── SVAMP/            

data process/


├── batch_arc.py      # Batch processing script for the ARC dataset

├── batch_asdiv.py    # Batch processing script for the ASDiv dataset

├── batch_gsm8k.py    # Batch processing script for the GSM8k dataset

├── batch_request.py  # Batch request script

├── batch_svamp.py    # Batch processing script for the SVAMP dataset

├── convert_to_list.py # Utility script to convert data to list format

├── dataloader.py     # Data loader for loading and preprocessing data

├── myloss.py         # Custom loss function(s)

├── readme.md         # This README file

├── split_val.py      # Script for splitting data into training and validation sets

├── spurious_token_game_3.py

└── test1_example.py  # Test or example script


## Main Scripts Explanation

* **`dataloader.py`**:
    * Function: Responsible for data loading, preprocessing, etc.
* **`batch_*.py` (e.g., `batch_arc.py`, `batch_gsm8k.py`)**:
    * Function: Performs batch processing operations for specific datasets (like ARC, GSM8k, etc.).
* **`myloss.py`**:
    * Function: Contains custom loss functions used in the project.
* **`convert_to_list.py`**:
    * Function: A utility tool for converting data from a certain format to a list format.
* **`split_val.py`**:
    * Function: Used to split the original dataset into training and validation sets for model evaluation.
* **`spurious_token_game_3.py`**:
    * Function: generate stg dataset
* **`test1_example.py`**:
    * Function: Provides an example of how to run or test some functionalities of the code.

## Dependencies

* Python 3.12.9
* Transformers

You can install dependencies using the following command:
```bash
pip install -r requirements.txt
```

## DEMO


![Alt text](image.png)

## STG
Machine learning theory posits that training and test sets are IID. However, due to the presence of spurious correlation, although the model's outcomes should be uniquely determined by causal features, the model may inadvertently capture these spurious correlations. This can lead to a reliance on spurious correlations rather than causal features when faced with a wide and diverse array of real-world scenarios, thereby compromising the model's reliability. So, our data generation processing follows the formalized expression below, where in the IID scenario, it satisfies:

$$ \mathcal{C}^s_i, I^s_i \sim Rand(1,10)$$
$$ \mathcal{S}^s_i = r_i*\mathcal{C}_i^s$$
$$ f(\mathcal{C}^s)=\sum_i k_i*\mathcal{C}^s_i$$
\[
\mathcal{A}^{s} = 
\begin{cases} 
High, &  f(\mathcal{C}^s) \geq \mu_h \\
Low, & else\\
\end{cases}
\]
where $ r_i,k_i,\mu_h $ are hyperparameters to control the ratio of high risk and low risk. The accuracy of random guessing is $50\%$.

In the OOD scenario, the three elements are independent of each other:
$$ \mathcal{S}^{ood}_i, \mathcal{C}^{ood}_i, I^{ood}_i \sim Rand(1,10)$$

A specific example is as follows:

![Alt text](image-2.png)

Specifically, the value of yellow fingers is 1.5 times that of smoking, the value of clothing size is the same as weight, and the value of hormones is 0.5 times that of exercise. All values are rounded down.
% \begin{align}
% f(\mathcal{C}^s) &= 1.2*\#Smoking+0.7*\#
%  Weight\\ \nonumber
%  &\quad -\#Exercise \nonumber
% \end{align}
$\mu_h=7.2$ and 
\begin{multline}
f(\mathcal{C}^s) = 1.2*\#Smoking+0.7*\#
 Weight\\ 
 -\#Exercise \nonumber
 \end{multline}
