import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from algorithms.utils.mixed_distances import huang_distance_matrix


#@profile
def laplacian_embedding(df, dim=None):
    """
    Compute Laplacian Eigenmaps embedding of a Mixed Dataset. To do so, we use Huang's distance, that is suited for mixed data.

    Parameters:
        df (pandas.DataFrame) : Dataset to compute Laplacian Eigenmaps from

    Return:
        embedding (numpy.array) : Projected coordinates in a Euclidean Space
    """

    if dim is None:
        dim = len(df.columns)

    t = len(df.columns)
    distances = huang_distance_matrix(df)
    embedding = laplacian_from_distance(distances, t, dim)
    
    return embedding

#@profile
def laplacian_from_distance(dist, t, dim):
    """
    Compute Laplacian Eigenmaps from a pairwise distance matrix. Adjacency matrix is computed using a Heat kernel.
    
    Parameters:
        dist (numpy.array) : Pairwise distances matrix to compute Laplacian Eigenmaps on.

        t (int) : Number of dimensions of the target space. Usually use the same dimensionality as the
        original dataset, as our objective is to translate mixed data in an Euclidean space and not to reduce
        its size. `t` is also used in the computation of the heat kernel (adjacency matrix).

    Return :
        embedding (numpy.array) : Computed coordinates with Laplacian Eigenmaps.

    """

    # Compute Adjacency Matrix from distances
    kernel = pd.DataFrame(dist).apply(lambda x: np.exp(-x/t))  

    # Spectral Embedding (using Laplacian Eigenmaps)
    embedding = SpectralEmbedding(dim,affinity="precomputed", gamma=4*t).fit_transform(kernel)

    return embedding


def lap_v2(df):

    distances = huang_distance_matrix(df)
    # Compute Adjacency Matrix from distances
    kernel = pd.DataFrame(distances).apply(lambda x: np.exp(-x/
        np.mean(sorted(x)[1:len(df.columns)]))
    )  

    # Spectral Embedding (using Laplacian Eigenmaps)
    embedding = SpectralEmbedding(len(df.columns),affinity="precomputed", gamma=4*len(df.columns)).fit_transform(kernel)

    return embedding