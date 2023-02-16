from algorithms.pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import umap.umap_ as umap
from scipy.spatial.distance import cdist

#@profile
def process(df, **kwargs):

    distances = huang_distance_matrix(df)

    n_components = np.min([len(df), len(df.columns)])
    
    embedding = UMAP_embedding(distances, n_components)

    clusters = pretopo_clustering(embedding)

    return clusters


#@profile
def huang_distance_matrix(df):
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    #n_components=min([categorical.shape[1],numerical.shape[1],len(df),numerical.shape[0], categorical.shape[0]])

    gamma = np.mean(np.std(numerical))/2
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma
    distances = np.nan_to_num(distances, posinf=99999999, neginf=-9999999) # Avoid crash on very large dimensions

    return distances


#@profile
def UMAP_embedding(dist, n_components):
    embedding = umap.UMAP(n_components=n_components,metric='precomputed').fit_transform(dist)
    return embedding

#@profile
def pretopo_clustering(data):
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(data))
    clusters = pretopo_clusters.astype(str)
    return clusters