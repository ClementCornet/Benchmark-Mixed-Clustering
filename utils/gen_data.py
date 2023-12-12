from sklearn.datasets import make_blobs # Generate blobs
import pandas as pd # Discretize categorical
from scipy.spatial.distance import cdist # Compute average distance
import numpy as np # Compute average distance

def generate_data(n_clusters=5,clust_std=0.1,n_num=15,n_cat=15,cat_unique=3,n_indiv=250):
    """
    Generates a Dataset to benchmark clustering algorithms. First generates centers for the given number of
    data points and features, then scales them to the average distance between centroids is 1. Then generates
    isotropic Gaussian blobs around those centers, and discretizes some variables to get categorical features.

    Parameters:
        n_clusters (int): Number of clusters to generate
        clust_std (float): Standard deviation of data around cluster centers
        n_num (int): Number of Numerical features
        n_cat (int): Number of Categorical features
        cat_unique (int): Number of unique values taken by each categorical feature
        n_indiv (int): Number of individuals (rows) to generate
    
    Returns:
        df (pandas.DataFrame): Generated Dataset
    """
    np.random.seed(1)
    # Generate Gaussian blobs
    mono_blobs = make_blobs(
        n_samples=n_clusters,
        n_features=n_num+n_cat,
        centers=n_clusters,
        return_centers=True,
        random_state=1
    )

    # Get Average distance between generated clusters centers
    distances = [cdist(mono_blobs[2],mono_blobs[2])[i][j] for i in range(n_clusters) for j in range(n_clusters) if i>j]
    avg_dist = np.mean(distances)

    # Generate new blobs, with clusters separated by 1
    generated_blobs = make_blobs(
        n_samples=n_indiv,
        n_features=n_num+n_cat,
        centers=mono_blobs[2]/avg_dist,
        cluster_std=clust_std,
        return_centers=True,
        random_state=1
    )

    # Only keep point coordinates
    df = pd.DataFrame(generated_blobs[0])

    # Discretize categorical variables
    for i in range(n_cat):
        df.iloc[:,-i-1] = pd.cut(df.iloc[:,-i-1],cat_unique,labels=False,
        ).astype(str)

    return df