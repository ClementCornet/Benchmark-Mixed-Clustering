from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import gower
from algorithms.utils.clustering_utils import elbow_method

#@profile
def process(df, **kwargs):
    """Agglomerative Hierarchical Clustering, using Gower's Distance, to get desired number of clusters"""

   # Get the number of clusters to process
    k = elbow_method(df)
    
    # Perfom Agglomerative Clustering
    clusters = phillip_ottaway(k,df)
    return clusters

##@profile
def get_gower_matrix(df):
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    categorical_indexes = []

    # Scaling
    scaler = StandardScaler()
    for c in categorical_columns:
        categorical_indexes.append(df.columns.get_loc(c))
    # create a copy of our data to be scaled
    df_scale = df.copy()
    # standard scale numerical features
    for c in numerical_columns:
        df_scale[c] = scaler.fit_transform(df[[c]])

    g_mat = gower.gower_matrix(df_scale) # Gower's pairwise Distances

    return g_mat

#@profile
def phillip_ottaway(k,df):
    g_mat = get_gower_matrix(df)
    # Perfom Agglomerative Clustering
    clusters = AgglomerativeClustering(k,affinity='precomputed',linkage='average').fit_predict(
        g_mat
    )
    return clusters