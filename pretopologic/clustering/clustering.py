from sklearn.base import BaseEstimator, ClusterMixin

class PretopoLogicCluster(BaseEstimator, ClusterMixin):
    """ Cluster build using pretopology"""

    def __init__(self, radius_coeff=20, threshold_coeff=0.1, closest_coeff=0.5, square_length_coeff=0.1, initial_set_size=1):
        self.radius_coeff = radius_coeff
        self.threshold_coeff = threshold_coeff
        self.closest_coeff = closest_coeff
        self.square_length_coeff = square_length_coeff
        self.initial_set_size = initial_set_size

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)


