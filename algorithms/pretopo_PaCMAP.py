from algorithms.pretopo.pretopo_base_clustering import pretopo_clustering
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding


@profile
def process(df, **kwargs):

    """
    Perform Pretopological Clustering on a mixed data set, in a tandem analysis with mixed PaCMAP.
    
    """

    embedding = PaCMAP_embedding(df)
    clusters = pretopo_clustering(embedding)

    return clusters