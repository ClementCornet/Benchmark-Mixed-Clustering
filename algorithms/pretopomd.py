import numpy as np

from scipy.spatial.distance import hamming, euclidean
from pretopologic.netgenerator.geometric import network_ball_distances, area
from pretopologic.space.pretopological_space import PretopologicalSpace, Prenetwork
from pretopologic.structure.closures import elementary_closures_shortest_degree
from pretopologic.structure.hierarchy import Pseudohierarchy


#@profile
def process(df, **kwarg):
    """Process K-Means of a Dataset's FAMD Coordinate"""
    radius_coeff = 20
    threshold_coeff = 2 # dois être superieur à 0
    closest_coeff = 1
    square_length_percentage = len(df)*(8/50)
    initial_sets_size = 1

    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns

    prenetwork_num = Prenetwork(df[numerical_columns], euclidean, area, network_ball_distances,
                         radius_coeff=radius_coeff, threshold_coeff=threshold_coeff, closest_coeff=closest_coeff,
                         square_length_coeff=square_length_percentage/100)
    prenetwork_cat = Prenetwork(df[categorical_columns], hamming, area, network_ball_distances,
                    radius_coeff=radius_coeff, threshold_coeff=threshold_coeff, closest_coeff=closest_coeff,
                    square_length_coeff=square_length_percentage/100)
    pre_space_num_cat = PretopologicalSpace([prenetwork_num, prenetwork_cat], [[0], [1]])
    closures_list_num_cat = elementary_closures_shortest_degree(pre_space_num_cat, initial_sets_size)
    pseudohierarchie_num_cat = Pseudohierarchy(closures_list_num_cat.copy(),)

    # name labels from the pseudohierarchy
    labels = np.zeros(len(df))
    for i, closure in enumerate(pseudohierarchie_num_cat.l_selected_sets):
        labels += (i + 1) * closure
        labels[labels > (i + 1)] = (i + 1)
    labels = labels - 1
    labels = labels.astype('int')
    unique_labels = list(set(labels))
    new_labels = [unique_labels.index(x) if x != -1 else x for x in labels]
    df['cluster'] = new_labels

    return new_labels

