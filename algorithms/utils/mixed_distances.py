import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

@profile
def huang_distance_matrix(df):
    """
    Get Pairwise distance matrix, using Huang's distance (suited for mixed data).

    Parameters:
        df (pandas.DataFrame) : DataFrame to compute the distance matrix
    
    Returns:
        distances (numpy.array) : Pairwise distance matrix

    """

    # Split Numerical and Categorical Variables, to compute Huang's distance
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0) # Scaling        
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    # Weight for categorical variables. Can vary, but 1/2 is the most common. (Used in kmodes.kprototypes code)
    gamma = np.mean(np.std(numerical))/2

    # Huang = Square Euclidean on Numerical + \gamma * Hamming for Categorical
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma

    # Very Large Dimension Datasets could return infinite values and cause crashes
    distances = np.nan_to_num(distances, posinf=99999999, neginf=-9999999)

    return distances