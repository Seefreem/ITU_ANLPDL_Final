# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.mixture import GaussianMixture
'''
https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_init.html#sphx-glr-auto-examples-mixture-plot-gmm-init-py
'''

def train_gmms(data, output_pdf, n_components_start=1, 
                n_components_end=15, n_components_step=2,
                output_dir='./cache/gmms', model_name=""):
    n_components = np.arange(n_components_start, 
                             n_components_end, 
                             n_components_step)
    bic_data = []
    aic_data = []
    min_aic = float('inf')
    min_bic = float('inf')
    min_aic_n = 0
    min_bic_n = 0 
    print('train_gmms')
    models = [None, None] # bic model and aic model
    for n in n_components:
        print('n_components', n)
        model = GaussianMixture(n, covariance_type='diag', random_state=0).fit(data)
        bic_data.append(model.bic(data))
        aic_data.append(model.aic(data))
        if bic_data[-1] < min_bic:
                min_bic = bic_data[-1]
                min_bic_n = n
                models[0] = model
        if aic_data[-1] < min_aic:
                min_aic = aic_data[-1]
                min_aic_n = n
                models[1] = model
        if not os.path.exists(output_dir+f'/log/'):
            os.makedirs(output_dir+f'/log/')
        save_gmms(output_dir+f'/log/{model_name}_n_{n}', model)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'/bic/'):
        os.makedirs(output_dir+'/bic/')
    if not os.path.exists(output_dir+'/aic/'):
        os.makedirs(output_dir+'/aic/')
    
    save_gmms(output_dir+f'/bic/best_{model_name}_n_{min_bic_n}', models[0])
    save_gmms(output_dir+f'/aic/best_{model_name}_n_{min_aic_n}', models[1])
    print(f'The best model in terms of BIC has {min_bic_n} components',
          f' in terms of AIC it has {min_aic_n} components')
    plt.plot(n_components, bic_data, label='BIC')
    plt.plot(n_components, aic_data, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    # Save the plot as a PDF file.
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    return models  # bic model and aic model

def save_gmms(gmm_name, gmm):
    # save to file
    np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
    np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
    np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)

def load_gmms(gmm_name):
    # reload
    means = np.load(gmm_name + '_means.npy')
    covar = np.load(gmm_name + '_covariances.npy')
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm