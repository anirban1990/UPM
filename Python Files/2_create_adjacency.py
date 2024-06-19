# %%
import pandas as pd
from scipy.spatial import distance
import numpy as np


# %%
# create adjacency matrix of 0s and 1s based on a Distance matrix 
# Distance matrix is generated from site locations (X and Y)
# Threshold distance should be set such that there is minimum 1 neighbor at each site 


#load the sample_dataset.csv file
df = pd.read_csv('sample_dataset.csv',sep=',',usecols=['X','Y','Z'])
N=df.shape[0]
xyz = df.to_numpy()

#Choose a threshold value
threshold = 1.1

# distance matrix
dist = distance.cdist(xyz,xyz,'euclidean')

# ignore diagonal values
np.fill_diagonal(dist, np.nan)

# extract i,j pairs where distance < threshold
paires = np.argwhere(dist<=threshold)

# groupby index
tmp = np.unique(paires[:, 0], return_index=True)
adj = np.split(paires[:,1], tmp[1])[1:]
adj = np.array(adj,dtype=object)

for i in range(len(adj)):
    for j in range(len(adj[i])):
        adj[i][j] = adj[i][j]  

adj_matrix = np.zeros((N, N), dtype="int32")
for area in range(N):
    adj_matrix[area, adj[area]] = 1

print(adj_matrix)

# %%
#save adjacency matrix as a .csv file

df = pd.DataFrame(adj_matrix)
df.to_csv("adjacency_matrix.csv", index=False,header=False)


