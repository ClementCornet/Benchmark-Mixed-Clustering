from pretopologic.constants import RADIUS_COEFFICIENT, THRESHOLD_COEFFICIENT, CLOSEST_COEFFICIENT, \
    SQUARE_LENGTH_COEFFICIENT
from pretopologic.netgenerator.geometric import areaminmax, area, diff, network_ball_distances
import numpy as np
import scipy as sp

# Here we define the class in charge of stocking the space and the method for the pseudoclure
# We need to define families of distance networks so the lsp can be more interesting
# We need to decide if we use prenetworks or neighborhoods
# dnf is a list of lists with integers between zero and len(neighborhoods)-1
class PretopologicalSpace:

    def __init__(self, prenetworks, dnf):
        """

        :param prenetworks: a list of prenetwork, each corresponding to a different characteristic of the elements
        :param dnf: the logical link between the criterias in the form of a list of list. If dnf == [[0,1],[2,3]]
        it means that for a point to be in the pseudoclosure of a set it needs to be close to it on the first AND the
        second parameter OR on the third AND the fourth parameter
        """
        self.prenetworks = prenetworks
        self.network_index = [i for i, prenetwork in enumerate(prenetworks) for item in prenetwork.thresholds]
        self.thresholds = [item for prenetwork in prenetworks for item in prenetwork.thresholds]
        self.dnf = dnf
        self.size = len(prenetworks[0].network) if prenetworks else 0


    def add_prenetworks(self, prenetworks):
        # we should check here the validity
        last_network_index = self.network_index[-1] if self.network_index else -1
        self.prenetworks += prenetworks
        self.network_index += [(i + last_network_index + 1) for i, prenetwork in enumerate(prenetworks)
                               for item in prenetwork.thresholds]
        self.thresholds += [item for prenetwork in prenetworks for item in prenetwork.thresholds]
        self.size = len(prenetworks[0].network) if not self.size else self.size

    def add_conjonction(self, conjonction):
        self.conjonctions.append(conjonction)


# Here we define the network_family class. This will allow us to faster calculate the pseudoclosure for a family
# of networks with the same weights but different thresholds, or with the same threshold and connections, but the
# weights are augmented or diminished by a constant factor.
# network is a list of numpy squared arrays, they all have to have the same size
# threshold is an array of floats, this way we can define many appartenence functions for a same network
class Prenetwork(object):

    def __init__(self, df_data, distance_calculation_method=sp.spatial.distance.euclidean,
                 area_calculation_method=areaminmax, formatting_network_method=network_ball_distances,
                 weights=[1],
                 radius_coeff=RADIUS_COEFFICIENT,
                 threshold_coeff=THRESHOLD_COEFFICIENT,
                 closest_coeff=CLOSEST_COEFFICIENT,
                 square_length_coeff=SQUARE_LENGTH_COEFFICIENT):
        """
        :param df_data: objects as a data frame with each row giving the values of one element
        :param distance_calculation_method: Method chosen to calculate the "distance" between two objects
        :param area_calculation_method: Method chosen to calculate the equivalent of the "area" occupied by the elements
        :param square_length_calculation_method: Method to calculate the average "length" of the "square" occupied by
                                                 each element
        :param formatting_network_method: Method used to determine a network such that the value of an edge are between
                                          0 and 1 and the value of an edge between two "close" points is close to 1 and
                                          the value of an edge between two "distant" points is close to 0 and the value
                                          of a loop is 0. Can approximate as 0 the value of "very distant" points.
        """
        self.df_data = df_data
        self.distance_method = distance_calculation_method
        self.area_method = area_calculation_method
        self.formatting_network_method = formatting_network_method
        self.weights = weights
        self._size = None
        self._dm = None
        self._area = None
        self._square_length = None
        self._closest = None
        self._inverse = None
        self._real_points = None
        self._radius = None
        self._thresholds = None
        self._formatted_network = None
        self._prenetwork = None
        self._radius_coeff = radius_coeff
        self._threshold_coeff = threshold_coeff
        self._closest_coeff = closest_coeff
        self._square_length_coeff = square_length_coeff

    @property
    def dm(self):
        """ Returns the distance matrix of the objects with the distance between 2 objects calculated using
            self.distance_method

        :return: The distance matrix of the objects with the distance between 2 objects calculated using
                 self.distance_method
        """
        if self._dm is None:
            self._dm = np.zeros(shape=(self.size, self.size), dtype=float)
            for i in range(self.size):
                for j in range(i + 1, self.size):
                    self.dm[i, j] = self.distance_method(self.df_data.iloc[i], self.df_data.iloc[j])
                    # print(f'the distance between {self.df_data.iloc[i][0]} and {self.df_data.iloc[j][0]} is {self.distance_method(self.df_data.iloc[i], self.df_data.iloc[j])}')
                    # print(f'self._dm[i, j]: {self._dm[i, j]}')
                    # print(f'self.dm[i, j]: {self.dm[i, j]}')
                # print(f'self.dm before for j: {self.dm}')
                for j in range(0, i):
                    self.dm[i, j] = self._dm[j, i]
                # print(f'self.dm after for j: {self.dm}')

        return self._dm

    @property
    def size(self):
        """ Returns the number of elements in self.df_data

        :return: The number of elements in self.df_data
        """
        if self._size is None:
            self._size = len(self.df_data)
        return self._size

    @property
    def area(self):
        """ Returns the "area" occupied by the elements

        :return: The "area" occupied by the elements
        """
        if self._area is None:
            self._area = self.area_method(self.dm, self.df_data)
        return self._area

    @property
    def square_length(self):
        """ Calculates the average "length" of the "square" occupied by each element, the higher the square length the
        less cluster are found, and the bigger they are.

        :return: The average "length" of the "square" occupied by each element
        """
        if self._square_length is None:
            self._square_length = np.sqrt(self.area / self.size) * self._square_length_coeff
        return self._square_length

    @property
    def closest(self):
        """ Calculates for each element the number of neighbors "close" to him, including himself

        :return: A list of size self.size such that self.closest[i] = the number of elements with a distance from the
                 element i lesser than self.quare_length/2. (counts i as a neighbor of i)
        """
        if self._closest is None:
            self._closest = np.sum(np.ma.masked_less(self.dm, self.square_length * self._closest_coeff).
                                   mask.astype('float'), axis=1)
        return self._closest

    @property
    def inverse(self):
        """ Calculates for each element the number (self.closest - 1) / self.closest. self.inverse[i] will be 0 for an
            elements with no close neighbor and will be close to 1 for an element with many neighbors.

        :return: A list of size self.size such that self.inverse[i] = (self.closest[i] - 1) / self.closest[i]
        """
        if self._inverse is None:
            self._inverse = (self.closest - 1) / self.closest
        return self._inverse

    @property
    def real_points(self):
        """ Calculates the number of elements minus the sum of the inverse. The more the elements are scattered the
            smaller the real_points value will be

        :return: self.size - sum(self.inverse)
        """
        if self._real_points is None:
            self._real_points = self.size - sum(self.inverse)
        return self._real_points

    @property
    def radius(self):
        """ Calculates the radius used by formatting_network

        :return: The radius used by formatting_network
        """
        if self._radius is None:
            self._radius = self._radius_coeff * self.square_length
        return self._radius

    @property
    def thresholds(self):
        """ Calculates the threshold used to create the prenetwork. The higher the threshold the more oultiers are found

        :return: The threshold used to create the prenetwork
        """
        if self._thresholds is None:
            self._thresholds = [(self.real_points/self.size)**(self._threshold_coeff)]
            # self._thresholds = [0.5]
        return self._thresholds

    @property
    def network(self):
        """ Creates the network using the formatting_network_method

        :return: The network using the formatting_network_method
        """
        if self._formatted_network is None:
            self._formatted_network = self.formatting_network_method(self.dm, self.radius)
        return self._formatted_network
