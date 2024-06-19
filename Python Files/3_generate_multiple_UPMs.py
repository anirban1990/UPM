# %%
# from signal import signal, SIGPIPE, SIG_DFL   
# signal(SIGPIPE,SIG_DFL) 

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
import pytensor.tensor as pt
import numpy.ma as ma
from scipy.stats import norm 
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})


# %%
#Read data from sample_dataset.csv

df = pd.read_csv('sample_dataset.csv')
Nsites=df.shape[0]
df.drop(['Sites','X','Y','Z'], axis=1, inplace=True)
data_source = df.to_numpy()

# %%
#Read data from adjacency_matrix.csv

df = pd.read_csv('adjacency_matrix.csv',header=None)
adj_matrix = df.to_numpy()


# %%
#Generate UPMs for multiple sample sizes 
Nsample=[10,50] #Nsample is an array denotiong the different sample sizes for which UPMs should be generated

global_posterior_mean = np.empty((Nsites,len(Nsample)))
global_posterior_std = np.empty((Nsites,len(Nsample)))

UPM_CAR = pm.distributions.multivariate.CARRV(ndims_params = [1, 2, 0, 1]) #Monkeypatching (source-code modification)
pm.CAR.rv_op = UPM_CAR #Monkeypatching (source-code modification)


for j in range(len(Nsample)):
    #prep the data
    data=data_source[:,:Nsample[j]]
    datatrans = data.transpose()


    # Bayesian modelling
    coords = {"Nsites":np.arange(Nsites)}
    with pm.Model(coords=coords) as model:
        sd = pm.Uniform("sd", lower=0, upper=5,dims="Nsites",initval=np.repeat(0.1,Nsites))
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1000,initval=100)
        C = pm.Uniform("C", lower=0.001, upper=1000,initval=1)
        phi= pm.CAR ("phi", mu=np.zeros(Nsites),tau=C*sd*sd, W=adj_matrix, alpha=0, dims="Nsites")
        mu = pm.Deterministic("mu", (beta0+phi))
        obs = pm.Normal("obs", mu=mu, sigma=sd, observed=datatrans)
        idata = pm.sample(draws=30000,tune=20000)


    #plot_Posterior
    stacked = az.extract(idata)
    Ypost_Mean = az.extract(idata, var_names="mu", num_samples=100)
    Ypost_STD = az.extract(idata, var_names="sd", num_samples=100)
    Posterior_Mean = Ypost_Mean.mean(axis=1) 
    Posterior_STD = Ypost_STD.mean(axis=1) 


    for i in range(len(Posterior_Mean)):
        current_mean = Posterior_Mean[i]
        current_std = Posterior_STD[i]
        std_threshold = 0.6
        if current_std < std_threshold:
            color = 'b'
            label = 'Low Variance'
        else:
            color = 'r'
            label = 'High Variance'
        x = np.arange(current_mean-10, current_mean+10, 0.05)
        y = norm.pdf(x, current_mean, current_std)
        plt.plot(x, y, color=color)
        plt.fill_between(x, y, color=color, alpha=0.5)


    # Naming the x-axis, y-axis and the whole graph 
    plt.xlabel("Data") 
    plt.ylabel("Probability density") 
    plt.title("Sample Size = {}".format(Nsample[j]))
    plt.xlim(0,14)
    plt.ylim(0,1.5)
    plt.xticks(np.arange(1, 14, step=1))
    
    # To load the display window 
    plt.savefig("UPM_N{}.png".format(Nsample[j]), dpi=300) 
    plt.close()



