import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, binom
import os

BETA_PRIOR_ALPHA = 2
BETA_PRIOR_BETA = 2

N_HEADS = 3
N_TAILS = 0

PATH_OUTPUT_FILES = "output"
FN_OUTPUT_FILE = "binom_lik_beta_prior.png"

if __name__=='__main__':
    comp_path_figure = os.path.join(PATH_OUTPUT_FILES,FN_OUTPUT_FILE)
    ## prepare for plotting (set plotting style,...)
    thetas = np.linspace(0.0,1.0,101)
    sns.set(style="whitegrid",rc={"grid.linewidth":0.2})
    sns.set_context("paper",font_scale=2.0)

    alpha_post = BETA_PRIOR_ALPHA + N_HEADS
    beta_post = BETA_PRIOR_BETA + N_TAILS

    post_mean = alpha_post/(alpha_post + beta_post)
    post_mode = (alpha_post-1)/(alpha_post+beta_post-2)
    post_median = (alpha_post - 1.0/3.0)/(alpha_post + beta_post - 2.0/3.0)
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(21,6))
    # 1st subplot: plot the prior
    sns.lineplot(thetas,beta.pdf(thetas,a=BETA_PRIOR_ALPHA,b=BETA_PRIOR_BETA),linewidth=3.5,ax=ax[0])
    ax[0].set_title("Prior Distribution")
    ax[0].set_xlabel("theta")
    ax[0].set_ylabel("probability density")
    ax[0].text(0.2,0.5,"alpha=2, beta=2")
    # 2nd subplot: plot the likelihood
    sns.lineplot(thetas,binom.pmf(N_HEADS,N_HEADS+N_TAILS,thetas),linewidth=3.5,ax=ax[1])
    ax[1].set_title("Likelihood function")
    ax[1].set_xlabel("theta")
    ax[1].set_ylabel("probability")
    ax[1].text(0.2,0.8,"N_heads=3,\nN_tails=0")
    # 3rd subplot: plot the posterior
    sns.lineplot(thetas,beta.pdf(thetas,a=alpha_post,b=beta_post),linewidth=3.5,ax=ax[2])
    theta_min = beta.ppf(0.05,a=alpha_post,b=beta_post)
    theta_max = beta.ppf(0.95,a=alpha_post,b=beta_post)
    thetas2 = np.linspace(theta_min,theta_max,100)
    ax[2].fill_between(thetas2,[0]*100,beta.pdf(thetas2,a=alpha_post,b=beta_post),alpha=0.5)

    sns.scatterplot([post_mode],[beta.pdf(post_mode,a=alpha_post,b=beta_post)],color="black",ax=ax[2],s=140)
    sns.scatterplot([post_mean],[beta.pdf(post_mean,a=alpha_post,b=beta_post)],color="black",ax=ax[2],s=140)
    sns.scatterplot([post_median],[beta.pdf(post_median,a=alpha_post,b=beta_post)],color="black",ax=ax[2],s=140)
    ax[2].set_title("Posterior distribution")
    ax[2].set_xlabel("theta")
    ax[2].set_ylabel("probability density")
    ax[2].text(0.05,2.0,"alpha=5, beta=2")
    fig.savefig(comp_path_figure)
    plt.close(fig)
