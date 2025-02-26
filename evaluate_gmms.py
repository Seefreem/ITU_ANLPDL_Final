import numpy as np
import os
from sklearn.mixture import GaussianMixture
import argparse
import os

from pprint import pprint
from manifold_learning_isomap_viz import load_judged_data
from linear_probe_utils import ProbeTrainer


def load_gmms(gmm_prefix):
    """
    Loads a pre-trained GMM (with 'diag' covariance type) from saved numpy files.
    It expects files with names:
      - {gmm_prefix}_means.npy
      - {gmm_prefix}_covariances.npy
      - {gmm_prefix}_weights.npy
    """
    means = np.load(gmm_prefix + '_means.npy')
    covar = np.load(gmm_prefix + '_covariances.npy')
    weights = np.load(gmm_prefix + '_weights.npy')
    
    n_components = len(means)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
    gmm.means_ = means
    gmm.covariances_ = covar
    gmm.weights_ = weights
    # For 'diag' covariance, the precision_cholesky is computed as the reciprocal square root of the covariance.
    # The shape of covar should be (n_components, n_features)
    gmm.precisions_cholesky_ = 1.0 / np.sqrt(covar)
    return gmm

def load_gmms_2(gmm_name):
    # reload
    means = np.load(gmm_name + '_means.npy')
    covar = np.load(gmm_name + '_covariances.npy')
    loaded_gmm = GaussianMixture(n_components = len(means), covariance_type='diag')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm

def compute_cluster_accuracies(gmm, data, labels):
    """
    Feeds the training set into the GMM, obtains clustering labels,
    and computes the accuracy for each cluster as:
         accuracy = (number of 1's in cluster) / (total samples in cluster)
    """
    cluster_labels = gmm.predict(data)
    # print('cluster_labels:\n', cluster_labels)
    print(cluster_labels.shape)
    
    unique_clusters = np.unique(cluster_labels)
    cluster_accuracies = {}

    for cluster in unique_clusters:
        idx = np.where(cluster_labels == cluster)[0]
        # print('idx', idx)
        # break
        if len(idx) == 0:
            cluster_accuracies[cluster] = 0
        else:
            # print('labels[idx]', labels[idx])
            cluster_accuracy = np.sum(labels[idx]) / len(idx)
            cluster_accuracies[cluster] = cluster_accuracy
    return cluster_accuracies

def predict_sample_probability(gmm, data, cluster_accuracies):
    """
    For each test sample, computes the probability of being labeled 1 using:
         p(sample is 1) = sum( cluster_prob[i] * cluster_accuracy[i] )
    where cluster_prob[i] is the probability that the sample belongs to cluster i.
    """
    # Predict probabilities for each sample over the clusters.
    probs = gmm.predict_proba(data)
    n_components = probs.shape[1]

    # Create an array of cluster accuracies ordered by cluster id.
    # It is assumed that clusters are labeled 0 to n_components-1.
    acc_array = np.array([cluster_accuracies.get(i, 0) for i in range(n_components)])
    # For each sample, compute weighted probability.
    sample_probabilities = np.dot(probs, acc_array)
    return sample_probabilities

def evaluate_gmms_models(model, train_data, train_labels, test_data, test_labels):
    # the best BIC model is models[0]
    print("Predicting clusters on the training set and calculating accuracies...")
    cluster_accuracies = compute_cluster_accuracies(model, train_data, train_labels)
    print("Predicting clustering probabilities on the test set...")
    sample_probabilities = predict_sample_probability(model, test_data, cluster_accuracies)
    metric = ProbeTrainer(None)
    return metric.get_acc(sample_probabilities, test_labels)


def main(args):
    # workspace
    workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, 'train', args.model)
    print('Layer', args.layer)
    print("Loading training set...")
    train_data, train_labels = load_judged_data(
        workspace_dir, layer=args.layer)
    
    if train_labels is None:
        raise ValueError("Training dataset must include binary labels.")

    print("Loading pre-trained GMM model (diag covariance)...")
    gmm_model_prefix = f'./cache/gmms/log/{os.path.basename(args.model)}_layer_{args.layer}__n_{args.n_component}'
    gmm = load_gmms(gmm_model_prefix)

    print("Loading test set...")
    workspace_dir = os.path.join(args.output_dir, 'answers', args.dataset, args.data_split, args.model)
    print('Layer', args.layer)
    test_data, test_labels = load_judged_data(
        workspace_dir, layer=args.layer)

    metric_scores = evaluate_gmms_models(gmm, train_data, train_labels, test_data, test_labels)
    print('Layer', args.layer, ". n_component", args.n_component)
    print(metric_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='coqa') # coqa, trivia_qa
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--judge_model", type=str, default='google/gemma-2-9b-it') # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--output_dir", type=str, default='./cache') # required=True, 
    parser.add_argument("--data_split", type=str, default='test') 
    parser.add_argument("--layer", type=int, default=11) 
    parser.add_argument("--n_component", type=int, default=60) 
    args = parser.parse_args()
    if 'gemma-2-2b-it' in args.model:
        layer_list       = [11, 13, 15, 17, 19, 21, 23, 25] # gemma 2 2B
        n_component_list = [60, 60, 40, 60, 60, 60, 30, 30]
    elif 'gemma-2-9b-it' in args.model:
        layer_list       = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41] # gemma 2 9B
        n_component_list = [60, 60, 60, 60, 60, 60, 60, 50, 50, 60, 40, 40, 40, 30, 20, 20]
    elif 'Llama-3.1-8B-Instruct' in args.model:
        layer_list       = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] # Llama-3.1-8B-Instruct
        n_component_list = [50, 50, 50, 40, 40, 40, 40, 40, 30, 30, 30]

    for layer_, component in zip(layer_list, n_component_list):
        args.layer = layer_
        args.n_component = component + 1
        print(f"\n\n ## args: {args} \n\n")
        main(args)

'''


'''
