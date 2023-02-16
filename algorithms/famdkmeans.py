import prince
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from algorithms.utils.clustering_utils import elbow_method


#@profile
def process(df, **kwargs):
    """Process K-Means of a Dataset's FAMD Coordinate"""

    # Get the number of clusters to process
    k = elbow_method(df)

    reduced = famd_embedding(df)

    # Process Standard K-Means
    km = KMeans(n_clusters=k)
    km.fit(reduced)
    return km.labels_

#@profile
def famd_embedding(df):
    # Get FAMD Coordinates
    famd = prince.FAMD(n_components=len(df.columns)) # Using the number of dimensions of the original data
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)
    return reduced