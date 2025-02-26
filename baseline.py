from linear_probe_utils import *
import glob
import argparse
import os
import pickle
import statistics

from tqdm import tqdm  # For a progress bar

parser = argparse.ArgumentParser(description='Template Label Counter')
parser.add_argument("--dataset",    type=str, action="store", default='trivia_qa') # coqa, trivia_qa
parser.add_argument("--model",      type=str, action="store", default='google/gemma-2-2b-it')
parser.add_argument("--judge_model",type=str, action="store", default='google/gemma-2-9b-it')
parser.add_argument("--output_dir", type=str, action="store", default='./cache') # required=True, 
parser.add_argument("--data_split", type=str, action="store", default='test') 
parser.add_argument("--f", type=str, action="store", default='') 

args = parser.parse_args()
print(args)

test_workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, 'test', args.model)
# Loading data
gt_label_dir  = os.path.join(test_workspace_dir, "judged_results")
rep_label_dir = test_workspace_dir
# Find all pickle files matching "result_*.pkl" inside save_dir.
gt_pickle_files = glob.glob(os.path.join(gt_label_dir, "result_*.pkl"))
rep_pickle_files = glob.glob(os.path.join(rep_label_dir, "result_*.pkl"))
gt_pickle_files.sort() 
rep_pickle_files.sort()

# print('pickle_files:', pickle_files)
if not rep_pickle_files:
    print(f"No pickle files found in {test_workspace_dir}.")
    
# Process each pickle file.
last_hidden_states_n = [] # layer 
gt_labels = []
avg_logprob = []
max_logprob = []
avg_entropy = []
max_entropy = []


for idx, file_path in tqdm(enumerate(zip(gt_pickle_files, rep_pickle_files))):
    # print('file_paths: ', file_path)
    if os.path.basename(file_path[0]) != os.path.basename(file_path[1]):
        print('ERROR!!!!! File name mismatched:', file_path)
        
    
    with open(file_path[0], "rb") as f: # Ground Truth
        result = pickle.load(f)
        gt_labels.append(1 if result['known'] == True else 0)
    # print('\n\n\n\n\n')
    with open(file_path[1], "rb") as f: # representations
        result = pickle.load(f)
        # print('generated_log_probs', len(result['generated_log_probs']))
        # print('entropy', len(result['entropy']))
        avg_logprob.append(statistics.mean([ -statistics.mean(sen) for sen in result['generated_log_probs']]))
        max_logprob.append(statistics.mean([ -max(sen) for sen in result['generated_log_probs']]))
        avg_entropy.append(statistics.mean([ -statistics.mean(sen) for sen in result['entropy']]))
        max_entropy.append(statistics.mean([ -max(sen) for sen in result['entropy']]))

from sklearn.metrics import roc_auc_score
print('avg_logprob', roc_auc_score(gt_labels, avg_logprob))
print('max_logprob', roc_auc_score(gt_labels, max_logprob))
print('avg_entropy', roc_auc_score(gt_labels, avg_entropy))
print('max_entropy', roc_auc_score(gt_labels, max_entropy))

'''
avg_logprob 0.728187134502924
max_logprob 0.7716179337231968
avg_entropy 0.728187134502924
max_entropy 0.7716179337231968

'''