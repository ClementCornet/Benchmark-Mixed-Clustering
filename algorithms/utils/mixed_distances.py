import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

#@profile
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


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

from rpy2.robjects.conversion import localconverter
import numpy as np
import pandas as pd

#@profile
def distance_matrix(df, metric='ahmad'):
    """
    Returns the pairwise distance matrix from a dataframe with mixed types.
    Uses distmix() from R package `kmed`.

    Parameters:
        df (pd.DataFrame): The DataFrame to process. Should contain both numerical and categorical features. 
        Some distances use 'binary' features, where some features only have 2 distinct values.

        metric (str): The distance to compute. Available distances are: Gower, Huang, Podani, 
        Harikumar, Wishart and Ahmad & Dey.

    Return:
        dist_matrix (numpy.array):  The pairwise distance of the dataframe
    """

    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('kmed')
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        kmed = importr('kmed')
        num_ids = []
        cat_ids = []
        bin_ids = []
        for i,col in enumerate(df.columns):
            if df[col].nunique() <= 2:
                bin_ids.append(i+1)
                continue
            elif np.issubdtype(df[col].dtype, np.number):
                num_ids.append(i+1)
                continue
            else:
                cat_ids.append(i+1)

        dist_matrix = kmed.distmix(
                                    data=df, 
                                    method=metric,  
                                    idbin=ro.r("NULL") if len(bin_ids)==0 else ro.IntVector(bin_ids),
                                    idnum=ro.r("NULL") if len(num_ids)==0 else ro.IntVector(num_ids),
                                    idcat=ro.r("NULL") if len(cat_ids)==0 else ro.IntVector(cat_ids)
                                )
    return dist_matrix