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
from manifold_learning_isomap_viz import isomap_dimen_redu, load_judged_data
from gmms import train_gmms
from evaluate_gmms import compute_cluster_accuracies, predict_sample_probability, evaluate_gmms_models
from linear_probe_utils import ProbeTrainer


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
    workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, args.data_split, args.model)
    
    if args.data_split == 'test' and args.dataset == 'coqa':
        args.data_split = 'validation'
    # download model
    print('#### load model')
    model, tokenizer = download_and_load_model(args.model, args.output_dir)
    model_config = model.config
    # generate answers: for each QA pair, generate 5 answers.
    # save the final answers and the last token representations from each layer.
    # Results are saved into files
    print("#### generate answers")
    generate_answers_and_save(dataset=dataset['train'],
                              model=model,
                              tokenizer=tokenizer,
                              output_dir=workspace_dir)

    # # LLM as a judge # Results are saved into files
    # print("#### assess generated answers")
    model, tokenizer = None, None
    model, tokenizer = download_and_load_model(args.judge_model, args.output_dir) 
    judge_answers_in_pickles(workspace_dir, model, tokenizer)
        
    print("#### visualize representations and train GMMs")
    test_workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, 'test', args.model)
    for i in range(model_config.num_hidden_layers-1, 10, -2):
        print('Layer', i)
        if not os.path.exists('./pics/'):
            os.makedirs('./pics/')
        last_hidden_states_n, gt_labels = load_judged_data(workspace_dir, i)  

        # train the GMMs model
        models, n_comp = train_gmms(last_hidden_states_n, 
                            "./pics/"+f'{os.path.basename(args.model)}_BIC_and_AIC_plot_layer_{i}.pdf',
                            n_components_start=1,
                            n_components_end=200,
                            n_components_step=20,
                            model_name=f'{os.path.basename(args.model)}_layer_{i}_')

        print("Loading test set...")
        print('Layer', i)
        test_data, test_labels = load_judged_data(
            test_workspace_dir, layer=i)
        metric_scores = evaluate_gmms_models(models[0], last_hidden_states_n, gt_labels, test_data, test_labels)
        print('Layer', i, ". n_component", n_comp)
        print(metric_scores)


    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='trivia_qa') # coqa, trivia_qa
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--judge_model", type=str, default='google/gemma-2-9b-it') # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--output_dir", type=str, default='./cache') # required=True, 
    parser.add_argument("--data_split", type=str, default='train') 
    
    args = parser.parse_args()
    print(f"\n\n ## args: {args} \n\n")
    main(args)

