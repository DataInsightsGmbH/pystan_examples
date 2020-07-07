import pandas as pd
import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
import os

PATH_STAN_MODELS = 'models'
FN_BLBP_MODEL = 'bin_lik_beta_prior.stan'
PATH_OUTPUT_FILES = 'output'

FN_STAN_OUTPUT = 'samples_bin_lik_beta_prior.csv'

FN_OUTPUT_FIGURE = "stan_samples_binom_lik_beta_prior.png"

# parameters for simulating data from a logistic regression model:
NUM_SAMPLES = 10000

if __name__=='__main__':
    data_for_stan = {'N_heads':3,
                     'N_tails':0}

    print(data_for_stan)

    comp_path_stan_model = os.path.join(PATH_STAN_MODELS,FN_BLBP_MODEL)

    with open(comp_path_stan_model,'rt') as sm_file:
        model_code_str = sm_file.read()

    print(model_code_str)
    stan_model =  pystan.StanModel(model_code=model_code_str)
    stan_fit   = stan_model.sampling(data=data_for_stan,
                             iter=2*NUM_SAMPLES,
                             chains=1
                            )

    print(stan_fit)
    thetas = np.array(stan_fit["theta"])
    comp_path_figure = os.path.join(PATH_OUTPUT_FILES,FN_OUTPUT_FIGURE)
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
    sns.set_context("paper", font_scale=2.0)
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(14,6))

    sns.scatterplot(range(len(thetas)),thetas,ax=ax[0],s=30)
    ax[0].set_title("Samples for theta")
    ax[0].set_xlabel("sample number")
    ax[0].set_ylabel("theta")

    sns.distplot(thetas,ax=ax[1],kde=False,hist=True,norm_hist=True)
    xs=np.linspace(0.0,1.0,101)
    sns.lineplot(xs,beta.pdf(xs,a=5,b=2),color="black")
    ax[1].set_title("Distribution of samples")
    ax[1].set_xlabel("theta")
    ax[1].set_ylabel("probability density")

    fig.savefig(comp_path_figure)
    plt.close(fig)
