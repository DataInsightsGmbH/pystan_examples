import pandas as pd
import numpy as np
import pystan
import os

PATH_STAN_MODELS = 'models'
FN_LR_MODEL = 'logistic_regression.stan'
PATH_STAN_OUTPUT = 'output'
FN_STAN_OUTPUT = 'samples_logistic_regression_300_data_points.csv'

# parameters for simulating data from a logistic regression model:
NUM_DATA_POINTS = 300
LOG_REG_A = 0.6
LOG_REG_B = -4.0
HOURS_STUDIED_MIN = 1.0
HOURS_STUDIED_MAX = 12.0
NUM_SAMPLES = 10000

def logistic_function(x,a,b):
    return 1.0/(1.0+np.exp(-(a*x+b)))

if __name__=='__main__':
    np.random.seed(1)
    hours = np.random.uniform(low=HOURS_STUDIED_MIN,high=HOURS_STUDIED_MAX,size=NUM_DATA_POINTS)
    probs = [logistic_function(hour,a=LOG_REG_A,b=LOG_REG_B) for hour in hours]
    passed = [np.random.binomial(n=1,p=p_i,size=1)[0] for p_i in probs]

    sim_data_dict = {"hours":hours,"probabilities":probs,"passed":passed}
    data_for_stan = {"N":NUM_DATA_POINTS,"x":hours,"y":passed}
    print(data_for_stan)
    # Compile the model
    comp_path_stan_model = os.path.join(PATH_STAN_MODELS,FN_LR_MODEL)

    with open(comp_path_stan_model,'rt') as sm_file:
        model_code_str = sm_file.read()

    print(model_code_str)
    stan_model =  pystan.StanModel(model_code=model_code_str)
    stan_fit   = stan_model.sampling(data=data_for_stan,
                             iter=2*NUM_SAMPLES,
                             chains=1
                            )
    print(stan_fit)
    df_stan_samples = pd.DataFrame({"alpha":list(stan_fit["alpha"]),"beta":list(stan_fit["beta"])})

    comp_path_samples = os.path.join(PATH_STAN_OUTPUT,FN_STAN_OUTPUT)
    df_stan_samples.to_csv(comp_path_samples)
