import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import torch
import pandas as pd
from scipy.spatial import distance
from time import time
from sklearn.manifold import (
    Isomap
)
from tqdm import tqdm  

def visualize_dataset(df, output_pdf):
    """
    Reads a CSV file with columns: x, y, label.
    Plots the points using a scatter plot with different colors for each label.
    Annotates each point with its label and saves the plot as a PDF.
    
    Parameters:
        df (dataframe): 
        output_pdf (str): Path where the PDF file will be saved.
    """
    # Create a new figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define markers and colors for each label.
    markers = {0: 'o', 1: 's'}
    colors = {0: 'blue', 1: 'red'}
    
    # Loop over the unique labels (0 and 1).
    for label in sorted(df['label'].unique()):
        # Filter the DataFrame for the current label.
        subset = df[df['label'] == label]
        ax.scatter(subset['x'], subset['y'],
                   label=f"Label {label}",
                   color=colors.get(label, 'black'),
                   marker=markers.get(label, 'o'),
                   alpha=0.5,
                   s=10)
        
        # # Annotate each point with its label.
        # for _, row in subset.iterrows():
        #     ax.annotate(str(label),
        #                 (row['x'], row['y']),
        #                 textcoords="offset points",
        #                 xytext=(5, 5),
        #                 ha="center",
        #                 fontsize=9)
    
    # Add labels, title, and legend to the plot.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dataset Visualization")
    ax.legend()
    ax.grid(True)
    
    # Save the plot as a PDF file.
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Plot saved as: {output_pdf}")



def train_dimen_redu_models(data, embeddings, y=None):
    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        print(f"Computing {name}...")
        start_time = time()
        # X {array-like, sparse matrix, BallTree, KDTree}
        # y Ignored. Not used, present for API consistency by convention.
        # Returns: X_new array-like, shape (n_samples, n_components)
        projections[name] = transformer.fit_transform(data, y) 
        timing[name] = time() - start_time
        print("Time:", timing[name])
    return projections, timing


def isomap_dimen_redu(data_dir, layer, n_neighbors, n_components=2):
    '''
    data_dir:
    n_neighbors: numbers of neighbors to calculate distance 
    n_components: target dimensions 
    '''
    # Loading data
    gt_label_dir  = os.path.join(data_dir, "judged_results")
    rep_label_dir = data_dir
    # Find all pickle files matching "result_*.pkl" inside save_dir.
    gt_pickle_files = glob.glob(os.path.join(gt_label_dir, "result_*.pkl"))
    rep_pickle_files = glob.glob(os.path.join(rep_label_dir, "result_*.pkl"))
    # print('pickle_files:', pickle_files)
    if not rep_pickle_files:
        print(f"No pickle files found in {data_dir}.")
        return
    # Process each pickle file.
    last_hidden_states_n = [] # layer 
    gt_labels = []
    gt_pickle_files.sort() 
    rep_pickle_files.sort()
    for idx, file_path in tqdm(enumerate(zip(gt_pickle_files, rep_pickle_files))):
        # print('file_paths: ', file_path)
        if os.path.basename(file_path[0]) != os.path.basename(file_path[1]):
            print('ERROR!!!!! File name mismatched:', file_path)
            return
        
        with open(file_path[0], "rb") as f: # Ground Truth
            result = pickle.load(f)
            gt_labels.append(1 if result['known'] == True else 0)
        # print('\n\n\n\n\n')
        with open(file_path[1], "rb") as f: # representations
            result = pickle.load(f)
            last_hidden_states_n.append(result['last_hidden_states'][layer])
        # break
    last_hidden_states_n = np.vstack(last_hidden_states_n)
    gt_labels = np.array(gt_labels).reshape((len(gt_labels), 1))
    print('last_hidden_states_n.shape', last_hidden_states_n.shape)
    print('gt_labels.shape', gt_labels.shape)
    # return 
    # explicit function to normalize array
    def normalize_2d(matrix):
        matrix = torch.from_numpy(matrix.astype(np.float16))
        matrix = torch.nn.functional.normalize(matrix)
        # norm = np.linalg.norm(matrix)
        # matrix = matrix/norm # normalized matrix
        return matrix.numpy()
    last_hidden_states_n = normalize_2d(last_hidden_states_n)
    # reduce the size
    idx_list = []
    # for i in range(len(last_hidden_states_n)):
    #     if len(idx_list) == 0:
    #         idx_list.append(i)
    #     else:
    #         flag = False
    #         for idx in range(len(idx_list)):
    #             # print(last_hidden_states_n[i, :], last_hidden_states_n[idx,:])
    #             # print(distance.cosine(last_hidden_states_n[i, :], last_hidden_states_n[idx,:]))
    #             if (distance.cosine(last_hidden_states_n[i, :], last_hidden_states_n[idx,:])) < 0.9:
    #                 print(distance.cosine(last_hidden_states_n[i, :], last_hidden_states_n[idx,:]))
    #                 flag = True
    #                 break
    #         if flag:
    #             idx_list.append(i)
    # print('Size: ', len(idx_list))
            
    
    # print(gt_labels)
    # print(last_hidden_states_n)
    # return last_hidden_states_n, gt_labels

    # Define the method for dimensionality reduction 
    embeddings = {
        "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=n_components),
    }

    # # dimensionality reduction
    # projections, timing = train_dimen_redu_models(last_hidden_states_n, embeddings)
    
    # # Visualization
    # for name in timing:
    #     title = f"{name} (time {timing[name]:.3f}s)"
    #     # plot_embedding(projections[name], title)
    #     # Create the DataFrame.
    #     df = pd.DataFrame({
    #         "x": projections[name][:, 0],
    #         "y": projections[name][:, 1],
    #         "label": gt_labels.ravel()  # Flatten gt_labels to shape (704,)
    #     })
    #     visualize_dataset(df, "./pics/"+title+f'_layer_{layer}_.pdf')
    # # plt.show()
    projections = []
    return last_hidden_states_n, gt_labels, projections

