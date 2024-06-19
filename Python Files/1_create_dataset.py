# %%
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import numpy.ma as ma
from scipy.stats import norm 
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})


# %%
#create sample_dataset

Nsites=13 # No. of sites
Nsites_SL= range(1,Nsites+1,1)
Nsites_X=[1,2,3,4,5,6,7,8,9,10,11,12,13] # X-coords of sites
Nsites_Y=[0,0,0,0,0,0,0,0,0,0,0,0,0] # Y-coords of sites
Nsites_Z=[0,0,0,0,0,0,0,0,0,0,0,0,0] # Y-coords of sites
NObs=200 # No. of observations at each site
mean=[1,2,3,4,5,6,7,8,9,10,11,12,13] # Known Mean of observations at each site
std=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,1,1,1,1,1,1] # Known Standard deviation of observations at each site 


np.random.seed(625)
data_source=np.empty((Nsites,NObs))
for i in range (0,Nsites):
    data_source[i,:]=np.random.normal(loc=mean[i],scale=std[i],size=NObs)



# %%
#save sample_dataset as a .csv file

df1 = pd.DataFrame(data_source)
df1.columns = ['Data_'+str(i) for i in range(1,NObs+1)]
df2 = pd.DataFrame({"Sites" : Nsites_SL, "X" : Nsites_X, "Y" : Nsites_Y, "Z" : Nsites_Z})
df = pd.concat([df2, df1], axis=1, join='inner')
df.to_csv("sample_dataset.csv", index=False)

# %%
#plot_sample_dataset

for i in range(len(mean)):
    current_mean = mean[i]
    current_std = std[i]
    std_threshold = 0.6
    if current_std < std_threshold:
        color = 'b'
        label = 'Low Variance'
    else:
        color = 'r'
        label = 'High Variance'
    x = np.arange(current_mean-10, current_mean+10, 0.05)
    y = norm.pdf(x, current_mean, current_std)
    plt.plot(x, y, color=color,label=label)
    plt.fill_between(x, y, color=color, alpha=0.5)


# Naming the x-axis, y-axis and the whole graph 
plt.xlabel("Data") 
plt.ylabel("Probability density") 
plt.xlim(0,14)
plt.xticks(np.arange(1, 14, step=1)) 
plt.ylim(0,1.5)
plt.yticks(np.arange(0, 1.55, step=0.25)) 
  
# Adding legend, which helps us recognize the curve according to it's color 
handles, labels=plt.gca().get_legend_handles_labels()
by_label=dict(zip(labels,handles))
plt.legend(by_label.values(), by_label.keys()) 
plt.savefig("sample_dataset.png",dpi=300) 
  
 



