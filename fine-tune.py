"""
    Fine tune GPT2 with custom corpus
    
    @link https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    
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

    def tokenize_function (examples):
        return tokenizer (examples["text"])


    # Texts are grouped together and chunk them in samples of length block_size. 
    # @todo You can skip that step if your dataset is composed of individual sentences.
    block_size = 128


    # @var huggingface_model String Load the base model
    huggingface_model = 'DeepESP/gpt2-spanish'


    # @var tokenizer AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained (huggingface_model, use_fast = True)
    
    
    # @var model AutoModelWithLMHead
    model = AutoModelWithLMHead.from_pretrained (huggingface_model)


    # @var training_args TrainingArguments
    training_args = transformers.TrainingArguments (
        output_dir = './results',
        evaluation_strategy = 'epoch',
        learning_rate = 2e-5,
        weight_decay = 0.01,
        logging_dir = './logs',
    )
    
    
    # @var datasets Dataset
    datasets = load_dataset ('csv', data_files = {'train': 'train.csv', 'validation': 'test.csv'})
    
    
    # Tokenize datasets
    datasets = datasets.map (tokenize_function, batched = True, remove_columns = ['text'])
    
    
    def group_texts (examples):
        """ 
        Main data processing function that will concatenate all texts from 
        our dataset and generate chunks of block_size.
        """
        
        # Concatenate all texts.
        concatenated_examples = {k: sum (examples[k], []) for k in examples.keys ()}
        
        
        # @var total_length
        total_length = len (concatenated_examples[list (examples.keys ())[0]])
        
        
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        
        
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range (0, total_length, block_size)]
            for k, t in concatenated_examples.items ()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    
    # Concatenate
    datasets = datasets.map (group_texts, batched = True)    
    
    
    
    # @var trainer
    trainer = transformers.Trainer (
        model = model, 
        args = training_args, 
        train_dataset = datasets['train'],
        eval_dataset = datasets['validation']
    )


    # Train
    trainer.train ()
    
    
    # Save model
    model.save_pretrained ('finetune')
    tokenizer.save_pretrained ('finetune')

        

if __name__ == "__main__":
    main ()