import numpy as np
import umap.umap_ as umap
from algorithms.utils.mixed_distances import huang_distance_matrix, distance_matrix



#@profile
def umap_embedding(df, n_components=None):
    """
    Compute UMAP embedding of a Mixed Dataset. To do so, we use Huang's distance, that is suited for mixed data.

    Parameters:
        df (pandas.DataFrame) : Dataset to compute UMAP from

    Return:
        embedding (numpy.array) : UMAP coordinates in a Euclidean Space
    """
    if n_components is None:
        n_components = np.min([len(df), len(df.columns)]) # Target Number of Coordinates

    


    distances = huang_distance_matrix(df)
    #distances = distance_matrix(df, 'wishart')
    embedding = umap_from_distance(distances, n_components)
    return embedding

#@profile
def umap_from_distance(dist, n_comp):
    """
    Compute UMAP embedding of a dataset, from a pairwise distances matrix. UMAP uses `dist` to construct its
    graph/adjacency matrix.
    
    Parameters:
        dist (numpy.array) : Pairwise distances matrix to compute the UMAP embedding from.

        n_comp (int) : Number of dimensions of the target space. Usually use the same dimmensionality as the
        original dataset, as our objective is to translate mixed data in an Euclidean space and not to reduce
        its size 

    Return :
        embedding (numpy.array) : Computed UMAP coordinates.

    """


    embedding = umap.UMAP(n_components=n_comp,metric='precomputed').fit_transform(dist)
    return embedding

