import torch
import numpy as np
from linear_probe_utils import *
import copy

import argparse
import os
import torch
import numpy as np 
import json

from models import download_and_load_model
from mlp_models import SAPLMANet, SimpleNet, DimMapNet
from manifold_learning_isomap_viz import load_judged_data  

parser = argparse.ArgumentParser(description='Template Label Counter')
parser.add_argument("--dataset",    type=str, action="store", default='trivia_qa') # coqa, trivia_qa
parser.add_argument("--model",      type=str, action="store", default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument("--judge_model",type=str, action="store", default='google/gemma-2-9b-it')
parser.add_argument("--output_dir", type=str, action="store", default='./cache') # required=True, 
parser.add_argument("--data_split", type=str, action="store", default='train') 
parser.add_argument("--f", type=str, action="store", default='') 

args = parser.parse_args()
print(args)

model, tokenizer = download_and_load_model(args.model, args.output_dir)


bs = 64
lr = 1e-4
epoch = 5
probe_trainer = ProbeTrainer(model.config,bs=bs,lr=lr,epoch=epoch)
model_config = model.config
model= None
tokenizer = None


# workspace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, 'train', args.model)
test_workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, 'test', args.model)
print(model_config.num_hidden_layers)
max_acc_hidden = []
probers = []
base_idx = 10
for layer in range(base_idx, model_config.num_hidden_layers):
    train_data, train_gt_labels = load_judged_data(workspace_dir, layer)
    unknown_data = train_data[train_gt_labels.reshape(-1) == 0.0, :].copy()
    unknown_labels = train_gt_labels[train_gt_labels.reshape(-1) == 0.0, :].copy()
    train_data = np.concatenate([train_data] + [unknown_data] * 5, axis=0)
    train_gt_labels = np.concatenate([train_gt_labels] + [unknown_labels] * 5, axis=0)
    train_data = torch.from_numpy(train_data).float().to(device)
    train_gt_labels = torch.from_numpy(train_gt_labels).float().to(device)

    test_data, test_gt_labels   = load_judged_data(test_workspace_dir, layer)
    test_data = torch.from_numpy(test_data).float().to(device)
    test_gt_labels = torch.from_numpy(test_gt_labels).float().to(device)
    # print(train_data.shape)
    state_dim = model_config.hidden_size
    # print(train_data.shape)
    # prober = DimMapNet(input_dim=state_dim).to(device)
    prober = SAPLMANet(input_dim=state_dim).to(device)
    
    optimizer = torch.optim.Adam(prober.parameters(), lr=probe_trainer.lr)

    test_results,prober = probe_trainer.fit_one_probe(prober, optimizer,
                                                      train_data, train_gt_labels,
                                                      test_data, test_gt_labels)
    max_acc = max(test_results,key=lambda x:x['auroc'])
    max_acc_hidden.append(max_acc)
    probers.append(copy.deepcopy(prober))
    print(layer, max_acc)


best_model_idx = max_acc_hidden.index(max(max_acc_hidden,key=lambda x:x['auroc']))
print("probe:",best_model_idx + base_idx, 
      max(max_acc_hidden,key=lambda x:x['auroc']))
print(probers[best_model_idx].parameters())
PATH = f'./cache/mlp_probers/{os.path.basename(args.model)}_layer_idx_{best_model_idx+base_idx}/'
if not os.path.exists(PATH):
    os.makedirs(PATH)
torch.save(probers[best_model_idx].state_dict(), PATH+'bset_model.pth')

with open(PATH + "log.json", "w") as json_file:
    json.dump(max_acc_hidden, json_file, indent=4)  # indent for better readability

'''
CoQA:
gemma 2 2B : probe: 18 {'acc': 0.742, 'precision': 0.931, 'recall': 0.736, 'specificity': 0.768, 'macro_f1': 0.676, 'auroc': 0.7521117608836906}
gemma 2 9B : probe: 27 {'acc': 0.798, 'precision': 0.973, 'recall': 0.8, 'specificity': 0.778, 'macro_f1': 0.644, 'auroc': 0.788888888888889}
llama: probe: 17 {'acc': 0.566, 'precision': 0.894, 'recall': 0.476, 'specificity': 0.833, 'macro_f1': 0.557, 'auroc': 0.6546345811051695}

TriviaQA:
gemma 2 2B : probe: 24 {'acc': 0.771, 'precision': 0.864, 'recall': 0.45, 'specificity': 0.958, 'macro_f1': 0.717, 'auroc': 0.7044740753987325}
gemma 2 9B : probe: 40 {'acc': 0.74, 'precision': 0.883, 'recall': 0.628, 'specificity': 0.889, 'macro_f1': 0.74, 'auroc': 0.7584618657589618}
llama: probe: 26 {'acc': 0.709, 'precision': 0.86, 'recall': 0.602, 'specificity': 0.861, 'macro_f1': 0.709, 'auroc': 0.7313087289812067}


'''