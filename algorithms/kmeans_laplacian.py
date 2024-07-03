import prince
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from algorithms.utils.clustering_utils import elbow_method
from algorithms.dimension_reduction.umap_reduction import umap_embedding



#@profile
def process(df, **kwargs):
    """Process K-Means of a Dataset's UMAP Coordinate"""

    # Get the number of clusters to process
    k = elbow_method(df)

    reduced = UMAP_embedding(df)

    # Process Standard K-Means
    km = KMeans(n_clusters=k)
    km.fit(reduced)
    print(km.labels_)
    return km.labels_
