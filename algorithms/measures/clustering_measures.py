from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.umap_reduction import umap_embedding

@profile
def internal_indices(df, clusters, **kwargs): 
    """
    Computes internal validation indices for a mixed clustering. Uses dimension reductions (FAMD, Laplacian Eigenmaps,
    UMAP and PaCMAP) to process Calinski-Harabasz, Davies-Bouldin and Silouhettes scores.

    Parameters:
        df (pandas.DataFrame) : Dataframe to process the indices on
        clusters (array_like) : Vector containing the results of a clustering
        keyword arguments : Possibility to pass the results of dimension reductions directly as kwargs, to avoid
            recomputing them. 'famd' to pass FAMD embedding, 'laplacian' for Laplacian Eigenmaps, 'pacmap' for PaCMAP
            and 'umap' for UMAP.

    Return:
        indices (dict) : Dictionnary, containing the indices for every dimension reduction.
            Ex : indices['FAMD']['CH'] to get Calinski-Harabasz score over FAMD coordinates.
    """

    # Get Dimension Reductions of `df`
    famd, lap, um, pm = None
    if 'famd' in kwargs.keys():
        famd =  kwargs['famd']
    else:
        famd = famd_embedding(df)

    if 'laplacian' in kwargs.keys():
        lap =  kwargs['laplacian']
    else:
        lap = laplacian_embedding(df)

    if 'pacmap' in kwargs.keys():
        pm =  kwargs['pacmap']
    else:
        pm = PaCMAP_embedding(df)

    if 'umap' in kwargs.keys():
        pm =  kwargs['umap']
    else:
        pm = umap_embedding(df)

    empty_dict = {'CH':0,'DB':-1,'Silhouette':0}
    indices = {
        'FAMD':empty_dict.copy(),
        'Laplacian':empty_dict.copy(),
        'UMAP':empty_dict.copy(),
        'PaCMAP':empty_dict.copy()
    }

    only_one_cluster = (len(set(clusters)) == 1)
    if only_one_cluster:
        return indices # No need to compute indices id there is only one cluster

    # Compute 3 indices for each dimension reduction
    indices['FAMD']['CH'] = calinski_harabasz_score(famd, clusters)
    indices['FAMD']['DB'] = davies_bouldin_score(famd, clusters)
    indices['FAMD']['Silhouette'] = silhouette_score(famd, clusters)

    indices['Laplacian']['CH'] = calinski_harabasz_score(lap, clusters)
    indices['Laplacian']['DB'] = davies_bouldin_score(lap, clusters)
    indices['Laplacian']['Silhouette'] = silhouette_score(lap, clusters)

    indices['UMAP']['CH'] = calinski_harabasz_score(um, clusters)
    indices['UMAP']['DB'] = davies_bouldin_score(um, clusters)
    indices['UMAP']['Silhouette'] = silhouette_score(um, clusters)

    indices['PaCMAP']['CH'] = calinski_harabasz_score(pm, clusters)
    indices['PaCMAP']['DB'] = davies_bouldin_score(pm, clusters)
    indices['PaCMAP']['Silhouette'] = silhouette_score(pm, clusters)

    return indices