from algorithms.pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import pacmap
import prince


#@profile
def process(df, **kwargs):

    n_components = np.min([len(df), len(df.columns)])

    famd_emb = famd_embedding(df)

    embedding = PaCMAP_embedding(famd_emb, n_components)
    
    clusters = pretopo_clustering(embedding)

    return clusters


#@profile
def famd_embedding(df):
    # Get FAMD Coordinates
    famd = prince.FAMD(n_components=len(df.columns)) # Using the number of dimensions of the original data
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)
    return reduced

#@profile
def PaCMAP_embedding(famd, n_comp):
    fit = pacmap.PaCMAP(n_components=n_comp,apply_pca=False).fit_transform(famd)
    pm = pd.DataFrame(fit)
    return pm


#@profile
def pretopo_clustering(data):
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(data))
    clusters = pretopo_clusters.astype(str)
    return clusters