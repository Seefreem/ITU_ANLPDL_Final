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
from manifold_learning_isomap_viz import isomap_dimen_redu
from gmms import train_gmms

def main(args):
    # download dataset
    print('#### load dataset')
    dataset = load_local_dataset(os.path.join(args.output_dir, args.dataset))
    print(dataset)
    if 'coqa' in args.dataset:
        dataset = flatten_coqa_dataset(dataset)
        print('After flattening: ', dataset, )
    pprint(dataset['train'][:2])
    # workspace
    workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, args.model)

    # download model
    print('#### load model')
    model, tokenizer = download_and_load_model(args.model, args.output_dir)

    # generate answers: for each QA pair, generate 5 answers.
    # save the final answers and the last token representations from each layer.
    # Results are saved into files
    print("#### generate answers")
    generate_answers_and_save(dataset=dataset['train'],
                              model=model,
                              tokenizer=tokenizer,
                              output_dir=workspace_dir)

    # LLM as a judge # Results are saved into files
    print("#### assess generated answers")
    model, tokenizer = None, None
    model, tokenizer = download_and_load_model(args.judge_model, args.output_dir) 
    judge_answers_in_pickles(workspace_dir, model, tokenizer)
        
    # visualize the representations. Graphs will be saved as PDFs
    print("#### visualize representations and train GMMs")
    for i in range(27):
        last_hidden_states_n, gt_labels, projections = isomap_dimen_redu(
            workspace_dir, layer=i, n_neighbors=10)

        # train the GMMs model
        models = train_gmms(last_hidden_states_n, 
                            "./pics/"+f'BIC_and_AIC_plot_layer_{i}.pdf',
                            n_components_start=10,
                            n_components_end=150,
                            n_components_step=10)



    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='coqa') # coqa, trivia_qa
    parser.add_argument("--model", type=str, default='google/gemma-2-2b-it')
    parser.add_argument("--judge_model", type=str, default='google/gemma-2-9b-it')
    parser.add_argument("--output_dir", type=str, default='./cache') # required=True, 
    print(f"\n\n ## args: {args} \n\n")
    args = parser.parse_args()
    main(args)

