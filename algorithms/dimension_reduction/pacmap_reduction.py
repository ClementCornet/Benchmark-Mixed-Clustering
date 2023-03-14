import pandas as pd
import numpy as np
import pacmap
from algorithms.dimension_reduction.famd_reduction import famd_embedding


#@profile
def PaCMAP_embedding(df, n_components=None):
    """
    Get PaCMAP (Factorial Analysis of Mixed Data) embedding of a given mixed dataset.
    Initialize with FAMD (instead of classical PCA, to adapt to mixed data), then process PaCMAP optimization.

    Parameters :
        df (pandas.DataFrame) : Mixed Dataset to process PaCMAP on
    
    Return:
        reduced (numpy.array) : PaCMAP coordinates of `df` in an Euclidean Space

    Note that the target space as the same number of dimensions as `df`, as the objective is to translate `df`
    in an Euclidean space more than reducing its size.
    
    """
    if n_components is None:
        n_components = np.min([len(df), len(df.columns)]) # Target Number of Coordinates

    famd = famd_embedding(df) # Initialize with FAMD

    embedding = PaCMAP_optimization(famd, n_components) # Optimization
    
    return embedding

#@profile
def PaCMAP_optimization(famd, n_comp):
    fit = pacmap.PaCMAP(n_components=n_comp,apply_pca=False).fit_transform(famd)
    emb = pd.DataFrame(fit)
    return emb