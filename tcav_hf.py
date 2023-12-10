# %%
import torch
import pandas as pd
import numpy as np

from torch import tensor
import torch.nn as nn
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from torch.utils.data import DataLoader, Dataset, IterableDataset

# %%
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
import jsonlines

import os
import matplotlib.pyplot as plt

# %%
class CustomTextDataset(IterableDataset):
        def __init__(self, filename, tokenizer, device, max_len = 32):
            self.df = pd.read_csv(filename)
            self.device = device
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __iter__(self):
            for _, row in self.df.iterrows():
                tokenized_text = self.tokenizer(row['text'], padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
                ret_tensor = tokenized_text['input_ids'].to(self.device)
                yield ret_tensor.squeeze(0)

# %%
def format_float(f):
        return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def assemble_concept(name, id, tokenizer, device, concepts_path="tutorials/data/tcav/text-sensitivity"):
        dataset = CustomTextDataset(concepts_path, tokenizer, device)
        concept_iter = DataLoader(dataset, batch_size=1)
        return Concept(id=id, name=name, data_iter=concept_iter)

# %%
from typing import Dict


from transformers.pipelines.base import GenericTensor


class CaptumPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        # `_legacy` is used to determine if we're running the naked pipeline and in backward
        # compatibility mode, or if running the pipeline with `pipeline(..., top_k=1)` we're running
        # the more natural result containing the list.
        # Default value before `set_parameters`

        outputs = model_outputs["logits"][0]

        return outputs
    
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        if isinstance(inputs, dict):
            return self.tokenizer(**inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        elif isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], list) and len(inputs[0]) == 2:
            # It used to be valid to use a list of list of list for text pairs, keeping this path for BC
            return self.tokenizer(
                text=inputs[0][0], text_pair=inputs[0][1], return_tensors=return_tensors, **tokenizer_kwargs
            )
        elif isinstance(inputs, list):
            # This is likely an invalid usage of the pipeline attempting to pass text pairs.
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
                ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
            )
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)

# %%
class TCAVTransformerPipeline():
    "Wrapper for Captum TCAV framework usage with Huggingface Pipeline"

    def __init__(self, name: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str):
        self.__name = name
        self.__model = model
        self.__tokenizer = tokenizer
        self.__device = device
        self.max_len = 32
        
    def apply_concept(self, text: str, concept_sets: list, out_file: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """

        inputs = self.generate_inputs(text)

        prediction = self.__model(inputs)
        
        # Tried initializing with model and forward_func, forward func doesn't work because direct access to model
        #layers is required. Model does not return logits.
        tcav = TCAV(self.__model, layers=['deberta.encoder.layer.0'])

        t = torch.argmax(prediction[0])

        # interpret expects logits as outputs, but hf models do not want to output logits directly
        positive_interpretations = tcav.interpret(
                                            inputs,
                                            experimental_sets=concept_sets,
                                            target=t
                                        )
        
        self.plot_tcav_scores(concept_sets, 
                              positive_interpretations, 
                              out_file=out_file, 
                              layers = ['deberta.encoder.layer']
                            )


    def plot_tcav_scores(self, experimental_sets, tcav_scores, out_file, layers = ['convs.2'], score_type='sign_count'):
        fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

        barWidth = 1 / (len(experimental_sets[0]) + 1)

        for idx_es, concepts in enumerate(experimental_sets):
            concepts = experimental_sets[idx_es]
            concepts_key = concepts_to_str(concepts)
            
            layers = tcav_scores[concepts_key].keys()
            pos = [np.arange(len(layers))]
            for i in range(1, len(concepts)):
                pos.append([(x + barWidth) for x in pos[i-1]])
            _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
            for i in range(len(concepts)):
                val = [format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]
                _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

            # Add xticks on the middle of the group bars
            _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
            _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
            _ax.set_xticklabels(layers, fontsize=16)

            # Create legend & Show graphic
            _ax.legend(fontsize=16)
        
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        plt.savefig(out_file)
        

    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        tok_text = self.__tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt").to(self.__device)
        return tok_text['input_ids']

# %%
# parser = argparse.ArgumentParser()
# parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
# parser.add_argument('--concept_dir', default='data', type=str, help='Directory where concept files are stored')
# parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
# parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
# parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
# parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
# args = parser.parse_args()

analysis_dir = os.getcwd() + '/tutorials/out/attributions/'
concept_dir = os.getcwd() + '/tutorials/data/tcav/text-sensitivity/'
model_checkpoint = 'microsoft/deberta-v3-base'
a1_analysis_file = os.getcwd() + '/tutorials/out/a1_analysis_data.jsonl'
num_labels = 2
output_dir = os.getcwd() + '/tutorials/out/checkpoints/'

# %%
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model.register_forward_hook(lambda self, input, output: output.logits)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#creating concepts, this code is functional
neutral_concept = assemble_concept('random_1', 0, concepts_path=concept_dir + '/neutral_samples.csv', tokenizer=tokenizer, device=device)
positive_concept = assemble_concept('random_2', 1, concepts_path=concept_dir +'/positive_samples.csv', tokenizer=tokenizer, device=device)

# %%
concepts = [[positive_concept, neutral_concept]]

#%%
tokenized_inp = tokenizer("I love you", return_tensors="pt")
tokenized_inp.to(device)
model.to(device)
model.eval()
outputs = model(**tokenized_inp)
print(outputs)

# %%
tcav_model = TCAVTransformerPipeline(name='tcav', model=model, tokenizer=tokenizer, device=device)

idx = 0
with jsonlines.open(a1_analysis_file, 'r') as reader:
    for obj in reader:
        tcav_model.apply_concept(obj["review"], concepts, os.path.join(output_dir, f'example_{idx}'))
# %%
