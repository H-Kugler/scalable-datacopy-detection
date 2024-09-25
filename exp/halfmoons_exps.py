import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from src.generative import MixedModel, Halfmoons, Mixture
from src.utils import run_exps


######### SET-UP #########

##########################
p = Halfmoons(noise=0.1).fit()  # true generative model
N_train = 2000  # number of samples for training distribution
N_test = 1000  # number of samples for generating and test distributions
# specify parameters for detectors
dcd_params = {
    "lmbda": np.arange(5, 19, 2),
    "gamma": [1 / 4000],
    "k": [1, 5, 10, 15],
}
tsd_params = {"num_regions": [5, 10, 20]}


######### RUN EXPERIMENTS  #########

####################################

######### True generating distribution, i.e. q=p
model_params = {"noise": [0.1]}  # no parameters to fit
run_exps(
    p=p,
    N_train=N_train,
    N_test=N_test,
    q=Halfmoons(),  # true generative model
    model_params=model_params,
    dcd_params=dcd_params,
    tsd_params=tsd_params,
    n_trials=100,
)


######### Mixed model, i.e. q = rho * q_copying + (1 - rho) * q_underfitting
model_params = {"rho": np.round(np.linspace(0.1, 0.9, 9), 1)}
run_exps(
    p=p,
    N_train=N_train,
    N_test=N_test,
    q=MixedModel(),
    model_params=model_params,
    dcd_params=dcd_params,
    tsd_params=tsd_params,
    n_trials=100,
)

######### Mixture model, i.e. q = rho * q_copying + (1 - rho) * p
model_params = {"rho": np.round(np.linspace(0.1, 0.9, 9), 1)}
run_exps(
    p=p,
    N_train=N_train,
    N_test=N_test,
    q=Mixture(rho=1, q1=MixedModel(rho=1), q2=p),
    model_params=model_params,
    dcd_params=dcd_params,
    tsd_params=tsd_params,
    n_trials=100,
)
