# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.mixture import GaussianMixture
'''
https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_init.html#sphx-glr-auto-examples-mixture-plot-gmm-init-py
'''

def train_gmms(data, output_pdf, n_components_start=1, n_components_end=15, n_components_step=2):
    n_components = np.arange(n_components_start, 
                             n_components_end, 
                             n_components_step)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data)
            for n in n_components]

    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    # Save the plot as a PDF file.
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    return models
