"""
    Testing GPT-2
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import os.path
import tensorflow as tf
import transformers
import pandas as pd
import torch

from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizer

from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
  

def main ():

    # @var huggingface_model String
    # huggingface_model = 'DeepESP/gpt2-spanish'
    huggingface_model = 'dccuchile/bert-base-spanish-wwm-cased'
    huggingface_model = 'finetune'


    # @var tokenizer AutoTokenizer
    tokenizer = BertTokenizer.from_pretrained (huggingface_model, do_lower_case=False)
    
    
    # @var model AutoModelWithLMHead
    model = BertForMaskedLM.from_pretrained (huggingface_model)
    
    
    # @var prefix_text
    text = "[CLS] Hacer una radiografía del [MASK] [SEP]"
    masked_indxs = [6]

    tokens = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    predictions = model(tokens_tensor)[0]

    for i,midx in enumerate(masked_indxs):
        idxs = torch.argsort(predictions[0,midx], descending=True)
        predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
        print('MASK',i,':',predicted_token)
        
    
    """
    for generated_text in generated_texts:
    
        # @var suggested_words List
        suggested_words = generated_text['generated_text'].split ()[num_words + 1: num_words + 3]
        print (" ".join (suggested_words))
    """

if __name__ == "__main__":
    main ()