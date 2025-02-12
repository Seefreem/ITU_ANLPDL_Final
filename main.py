import argparse
import os
import pickle
import torch
import numpy as np 


from tqdm import tqdm  # For a progress bar
from pprint import pprint
from download_dataset import prepare_coqa, load_local_dataset, flatten_coqa_dataset
from models import download_and_load_model
from generate_answers import generate_answers_and_save, judge_answers_in_pickles

def main(args):
    # download dataset
    dataset = load_local_dataset(os.path.join(args.output_dir, args.dataset))
    print(dataset)
    pprint(dataset['train'][0])
    dataset = flatten_coqa_dataset(dataset)
    print('After flattening: ', dataset, )
    pprint(dataset['train'][:2])

    # download model
    model, tokenizer = download_and_load_model(args.model, args.output_dir)

    # generate answers: for each QA pair, generate 5 answers.
    save the final answers and the last token representations from each layer.
    Results are saved into files
    generate_answers_and_save(dataset=dataset['train'],
                              model=model,
                              tokenizer=tokenizer,
                              output_dir=os.path.join(args.output_dir, 'answers', args.dataset, args.model))

    # LLM as a judge # Results are saved into files
    model, tokenizer = None, None
    model, tokenizer = download_and_load_model(args.judge_model, args.output_dir) 
    judge_answers_in_pickles(os.path.join(args.output_dir, 'answers', args.dataset, args.model),
                            model, tokenizer)
        

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='coqa')
    parser.add_argument("--model", type=str, default='google/gemma-2-2b-it')
    parser.add_argument("--judge_model", type=str, default='google/gemma-2-9b-it')
    parser.add_argument("--output_dir", type=str, default='./cache') # required=True, 
    args = parser.parse_args()
    main(args)

