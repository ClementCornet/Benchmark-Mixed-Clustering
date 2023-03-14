import gower
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

#@profile
def elbow_method(df):
    """
    Performs Elbow Method to determine the optimal number of clusters in a dataframe. To fit mixed data,
    Gower's distance matrix is used. The metric used as scoring parameter is the Calinski-Harabasz index.

    Parameters:
        df (pandas.DataFrame): DataFrame to perform the Elbow Method on

    Returns:
        k (int): Optimal number of clusters in df
    
    """
    g_mat = gower.gower_matrix(df)

    model = KElbowVisualizer(KMeans(), k=10, distance_metric='precomputed',metric='calinski_harabasz')

    model.fit(g_mat)
    
    return model.elbow_value_