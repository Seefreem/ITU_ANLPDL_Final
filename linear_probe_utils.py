import torch
import torch.nn as nn
# from baukit import Trace, TraceDict
from sklearn.metrics import confusion_matrix
import copy
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

very_small_value = 0.000001

class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int,):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.loss = nn.BCELoss()
        
    def forward(self,state, label):
        out = torch.sigmoid(self.linear(state))
        loss = self.loss(out,label)
        return out, loss
    
class ProbeTrainer():
    def __init__(self, model_config, epoch=10, lr=0.01, bs=32):
        self.model_config = model_config
        self.epoch = epoch
        self.lr = lr
        self.bs = bs

    def get_acc(self,score,label):
        predict = torch.Tensor([s>0.5 for s in score])
        label = torch.Tensor(label.squeeze())
        acc = sum(predict==label)/len(label)
        tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
        precision = tp/(tp+fp+very_small_value)
        recall = tp/(tp+fn+very_small_value)
        specificity = tn/(tn+fp+very_small_value)
        macro_f1 = f1_score(label, predict, average='macro')
        # print(label.shape, predict.shape)
        auroc = roc_auc_score(label, predict)

        result = {"acc":round(acc.item(),3),
                  "precision":round(precision,3), 
                  "recall":round(recall,3),
                  "specificity":round(specificity,3),
                  'macro_f1': round(macro_f1,3),
                  'auroc': auroc}

        return result

    def fit_one_probe(self, prober, optimizer,
                      train, train_labels,
                      test,  test_labels):
        best_probe = None
        best_auroc = 0
        test_results = []
        train_index = [i for i in range(len(train))] 
        test_index  = [i for i in range(len(test))]
        random.shuffle(train_index)
        random.shuffle(test_index)
        train_index = np.array(train_index)
        test_index = np.array(test_index)

        test_cout = int(len(train)/10)
        for ep in range(self.epoch):
            for i in range(0,len(train),self.bs):
                optimizer.zero_grad()
                state = train[train_index[i:i+self.bs], :]
                label = train_labels[train_index[i:i+self.bs]]

                out,loss = prober(state,label)
                loss.backward()
                optimizer.step()
                if i%test_cout==0:
                    prober.eval()
                    state = test
                    label = test_labels
                    out,loss = prober(state,label)
                    result = self.get_acc(out, label.detach().cpu())
                    if result["auroc"]>best_auroc:
                        best_probe = copy.deepcopy(prober)
                        best_auroc=result["auroc"]
                    test_results.append(result)
                    # print(result)
                    prober.train()
        return test_results, best_probe
