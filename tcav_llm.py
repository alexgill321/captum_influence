import torch
import pandas as pd
import numpy as np

from torch import tensor
import torch.nn as nn
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from datasets import load_dataset

# %%
from captum.concept import TCAVLM
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
import jsonlines

import os
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# %%
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m")
model.to(device)

# %%

inputs = tokenizer("Hello, I am", return_tensors="pt").to(device)
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])

# %%
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

#%%
print(model.config)
print(model.config.max_position_embeddings)

# %%
for i, example in enumerate(dataset):
    inputs = tokenizer.encode(example["text"], max_length=model.config.max_position_embeddings-200, return_tensors="pt").to(device)
    tokens = model(inputs)
    print(tokens.logits.shape)
    for j in range(tokens.logits.shape[1]):
        print(tokenizer.decode(torch.argmax(tokens.logits[0, j, :])))
    break

# %%
def preprocess_function(examples):
    tokenized_input = tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=model.config.max_position_embeddings, 
                return_tensors="pt"
            )
    input_ids = tokenized_input["input_ids"]
    return input_ids.to(device)

class PreprocessedDataset(IterableDataset):
    def __init__(self, dataset, preprocess_function):
        self.dataset = dataset
        self.preprocess_function = preprocess_function

    def __iter__(self):
        for example in self.dataset:
            yield self.preprocess_function(example)

    def take(self, n):
        return PreprocessedDataset(self.dataset.take(n), preprocess_function)

    
    def skip(self, n):
        return PreprocessedDataset(self.dataset.skip(n), preprocess_function)

tokenized_data = PreprocessedDataset(dataset, preprocess_function)

# # %%
# for i, example in enumerate(tokenized_data):
#     tokens = model(example['input_ids'])
#     print(tokens.logits.shape)
#     for j in range(tokens.logits.shape[1]):
#         print(tokenizer.decode(torch.argmax(tokens.logits[0, j, :])))

# # %%
# class CustomTextDataset(IterableDataset):
#         def __init__(self, dataset, tokenizer, device, max_len = 32):
#             self.df = dataset
#             self.device = device
#             self.tokenizer = tokenizer
#             self.max_len = max_len
            
#         def __iter__(self):
#             for _, row in self.df.iterrows():
#                 tokenized_text = self.tokenizer(row['text'], padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
#                 ret_tensor = tokenized_text['input_ids'].to(self.device)
#                 yield ret_tensor.squeeze(0)

# %%
concept_data_1 = tokenized_data.take(100)
other_data = tokenized_data.skip(100)
concept_data_2 = other_data.take(100)

# # %%
# model.to(device)
# for i, example in enumerate(concept_data_1):
#     tokens = model(example)
#     print(tokens.logits.shape)
#     if i > 2:
#         break  

# # %% Retrieve the output for the last token in the ouput sequence
# # concept_hook = model.register_forward_hook(lambda self, input, output: output.logits[:, -1, :])
# concept_hook = model.register_forward_hook(lambda self, input, output: output.logits.squeeze(0).argmax(dim=-1, keepdim=False))
# for i, example in enumerate(concept_data_2):
#     tokens = model(example)
#     print(tokens.shape)
#     if i > 2:
#         break

# concept_hook.remove()

# # %%
# model.to(device)
# for i, example in enumerate(concept_data_1):
#     tokens = model(example)
#     print(tokens.logits.shape)
#     if i > 2:
#         break  
# Generate new datasets using the tokenized data?
# %% Create Concepts
random_concept_1 = Concept(id=0, name="random_concept_1", data_iter=concept_data_1)
random_concept_2 = Concept(id=1, name="random_concept_2", data_iter=concept_data_2)

concepts = [[random_concept_1, random_concept_2]]
#%%
for module in model.modules():
    print(module)
#%% Generate Queries and Labels
tokenized_dataset = tokenized_data.skip(100)
input_dataset = tokenized_dataset.take(10)

tcav = TCAVLM(model, layers=["gpt_neox.layers.1.mlp.dense_h_to_4h"])

#%%
for example in input_dataset:
    tcav.interpret(
        example,
        concepts
    )
# %%
