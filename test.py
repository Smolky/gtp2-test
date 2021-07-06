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

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead

from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
  

def main ():

    # @var huggingface_model String
    huggingface_model = 'DeepESP/gpt2-spanish'
    huggingface_model = 'finetune'


    # @var tokenizer AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained (huggingface_model, use_fast=True)
    
    
    # @var model AutoModelWithLMHead
    model = AutoModelWithLMHead.from_pretrained (huggingface_model)

    
    # @var text_generation
    text_generation = pipeline ('text-generation', model = model, tokenizer = tokenizer)
    
    
    # @var prefix_text
    prefix_text = "El mundo es un"
    
    
    # @var num_words int
    num_words = len (prefix_text.split ())
    
    
    # @var generated_texts List
    generated_texts = text_generation (prefix_text, max_length = 50, do_sample = False)
    
    
    for generated_text in generated_texts:
    
        # @var suggested_words List
        suggested_words = generated_text['generated_text'].split ()[num_words + 1: num_words + 3]
        print (" ".join (suggested_words))
    

if __name__ == "__main__":
    main ()