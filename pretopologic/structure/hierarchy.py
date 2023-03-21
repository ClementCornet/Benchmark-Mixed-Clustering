import numpy as np
import random
import itertools
import logging

__author__ = "equipeai: ai@energisme.com, jbosom: jeremie.bosom@energisme.com,"

log = logging.getLogger(__name__)

def seti_in_setj(setx, sety):
    intersection = np.dot(setx, sety.T)
    # log.debug("intersection :", intersection)
    # log.debug(sum(setx))
    # log.debug(sum(sety))
    # log.debug(int(intersection == sum(setx)))
    return int(intersection == sum(setx))


class Pseudohierarchy:
    
    def __init__(self, elementary_cls_original, attraction_threshold=0.5):
        self._elementary_cls = elementary_cls_original
        self.attraction_threshold = attraction_threshold
        self._initial_sets_nb = None
        self._attraction_matrix = None
        self._adjacency_matrix_with_equivalent = None
        self._l_selected_sets = None
        self._adjacency_matrix = None
        self._clusters = None

    @property
    def initial_sets_nb(self):
        """ Original number of sets

            :returns: The number of sets given in parameters
            :rtype: int
        """
        if self._initial_sets_nb is None:
            self._initial_sets_nb = len(self._elementary_cls)
        return self._initial_sets_nb

    @property
    def attraction_matrix(self):
        """ Determines the force of attraction of one set on another set for each pair of set in 
            elementary_cls_original.
            The force of attraction of a set A on a set B is proportional to |A|/|B| and |A n B|/|A|

            :returns: A matrix representing the attraction of the sets such that attraction_matrix[i, j] represents how
            much i is attracted to j
            :rtype: numpy.ndarray
        """
        # We should try to vectorize this
        # todo: Not calculate the attraction of a closure on itself (it is useless)
        if self._attraction_matrix is None:    
            elementary_cls = np.array(self._elementary_cls.copy())
            self._attraction_matrix = np.zeros((self.initial_sets_nb, self.initial_sets_nb))
            log.debug(len(elementary_cls))
            log.debug(self.initial_sets_nb)
            for i, cls in enumerate(elementary_cls):
                # log.debug('i: ', i)
                # log.debug('closure: ', cls)
                temporary_cls = elementary_cls[i:]
                # log.debug('temporary closure: ', temporary_cls)
                # log.debug('temporary closure.T: ', temporary_cls.T)
    
                intersections = np.dot(cls, temporary_cls.T)
                # log.debug('intersection ', intersections)
                size_temporary_closures = np.sum(temporary_cls, axis=1)
                # log.debug('size_temporary_closures: ', size_temporary_closures)
    
                # Part of temporary_closure withing closure
                part_of_tmp_cls_within_cls = intersections / size_temporary_closures
                # log.debug('part_of_tmp_cls_within_cls: ', part_of_tmp_cls_within_cls)
    
                # Part of temporary_closure withing closure
                part_of_cls_within_tmp_cls = intersections / sum(cls)
                # log.debug('part_of_cls_within_tmp_cls: ', part_of_cls_within_tmp_cls)
    
                # How big is cls compared to tmp_cls ?
                cls_on_tmp_cls = sum(cls) / size_temporary_closures
    
                # How big is cls compared to tmp_cls ?
                tmp_cls_on_cls = size_temporary_closures / sum(cls)
    
                # An arrow from cls to tmp_cls weights the attraction that tmp_cls has over cls.
                self._attraction_matrix[i, i:] = tmp_cls_on_cls * part_of_cls_within_tmp_cls
                self._attraction_matrix[i:, i] = cls_on_tmp_cls * part_of_tmp_cls_within_cls
        # log.debug('self._attraction_matrix: ', self._attraction_matrix)
        return self._attraction_matrix

    @property
    def adjacency_matrix_with_equivalent(self):
        """ Creates the matrix of adjacency taking into account the attraction between the sets and the inclusions

            :returns: A matrix representing the relationship between the sets:
            adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 1 means i and j are equivalent
            adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 0 means i is a child of j
            adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 1 means i and j have no relationship
            :rtype: numpy.ndarray
        """
        if self._adjacency_matrix_with_equivalent is None:
            # First we determine if there is an attraction stronger than the threshold between each cluster.
            self._adjacency_matrix_with_equivalent = np.zeros((self.initial_sets_nb, self.initial_sets_nb), dtype=int)
            self._adjacency_matrix_with_equivalent[self.attraction_matrix >= self.attraction_threshold] = 1
            # Then we assure that if a set I is in a set J then I is a "child" of J and cannot be a "parent" of
            # J (If i is in j then there is adjacency from i to j and there is no adjacency from j to i)
            for i in range(self.initial_sets_nb):
                for j in range(self.initial_sets_nb):
                    if i != j:
                        if seti_in_setj(self._elementary_cls[i], self._elementary_cls[j]) == 1:
                            self._adjacency_matrix_with_equivalent[i, j] = 1
                        if seti_in_setj(self._elementary_cls[j], self._elementary_cls[i]) == 1:
                            self._adjacency_matrix_with_equivalent[i, j] = 0
        return self._adjacency_matrix_with_equivalent

    @property
    def adjacency_matrix(self):
        """ Creates the matrix of adjacency taking into account the attraction between the sets and the inclusions

        :returns: A matrix representing the relationship between the sets:
        adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 0 means i is a child of j
        adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 1 means i and j have no relationship
        :rtype: numpy.ndarray
        """
        if self._adjacency_matrix is None:
            self.hierarchy()
        return self._adjacency_matrix

    def hierarchy(self):
        """ Determines the final sets and the final adjacency matrix by keeping only one set in each group of
            equivalent sets """
        equivalent_set_matrix = self.adjacency_matrix_with_equivalent * self.adjacency_matrix_with_equivalent.T
        l_l_equivalents = [list(np.argwhere(i == 1).flatten()) for i in equivalent_set_matrix]
        log.debug('l_l_equivalents: ', l_l_equivalents)
        l_l_equivalents.sort()
        l_l_equivalents = list(k for k, _ in itertools.groupby(l_l_equivalents))  # Delete the duplicates
        log.debug('l_l_equivalents: ', l_l_equivalents)
        s_not_selected = set()
        # We are going to pick only one of each group of equivalent set.
        for l_equivalents in l_l_equivalents:
            log.debug("l_equivalents: ", l_equivalents)
            s_not_selected.update(l_equivalents[1:])
            log.debug("not selected: ", s_not_selected)
        log.debug(s_not_selected)
        l_representatives = [i for i in range(len(self.adjacency_matrix_with_equivalent)) if i not in s_not_selected]
        log.debug('l_representatives: ', l_representatives)
        # log.debug("adjacency_matrix_with_equivalent: ", adjacency_matrix_with_equivalent)
        self._adjacency_matrix = self._adjacency_matrix_with_equivalent[l_representatives, :][:, l_representatives]
        log.debug("adj:\n", self._adjacency_matrix)
        self._l_selected_sets = [x for i, x in enumerate(self._elementary_cls) if i in l_representatives]
        log.debug("l_selected_closures: ", self._l_selected_sets)
        log.debug("l_selected_closures size :", [sum(x) for i, x in enumerate(self._elementary_cls) if i in l_representatives])

    @property
    def clusters(self):
        """ Creates the clusters """
        # todo: If one closure is included inside an other then it should be a child of that closure
        # First we determine if there is an attraction stronger than the threshold between each cluster.
        # This gives the matrix adjacency_matrix_with_equivalent
        adjacency_matrix_with_equivalent = np.zeros((self.initial_sets_nb, self.initial_sets_nb), dtype=int)
        adjacency_matrix_with_equivalent[self.attraction_matrix >= self.attraction_threshold] = 1
        log.debug("adjacency_matrix_with_equivalent: ", adjacency_matrix_with_equivalent)
        log.debug("self._elementary_cls: ", self._elementary_cls)
        # log.debug("adjacency_matrix_with_equivalent: ", adjacency_matrix_with_equivalent)
        # If for 2 sets A and B we see that A attracts B and B attracts A then they are equivalent and we must only keep
        # only one of them for our clusterisation
        equivalent_set_matrix = adjacency_matrix_with_equivalent * adjacency_matrix_with_equivalent.T
        log.debug("equivalent_set_matrix: ", equivalent_set_matrix)
        # log.debug("equivalent_set_matrix: ", equivalent_set_matrix)
        l_l_equivalents = [list(np.argwhere(i == 1).flatten()) for i in equivalent_set_matrix]
        import itertools
        l_l_equivalents.sort()
        l_l_equivalents = list(k for k, _ in itertools.groupby(l_l_equivalents))
        log.debug("l_l_equivalents : ", l_l_equivalents)
        l_representatives = list()
        set_done = set()
        # We are going to pick only one of each group of equivalent set randomly. A set which is strictly included in
        # an other equivalent set cannot be chosen.
        for l_equivalents in l_l_equivalents:
            log.debug("l_equivalents: ", l_equivalents)
            log.debug("set_done: ", set_done)
            eq_set = {i for i in l_equivalents
                       for j in l_equivalents
                       if i != j
                       if seti_in_setj(self._elementary_cls[i], self._elementary_cls[j]) != 1
                      } if len(l_equivalents) > 1 else set(l_equivalents)
            # Randomly choose a set in those not already chosen
            l_non_selected_representatives = list(eq_set.difference(l_representatives))
            if l_non_selected_representatives:
                l_representatives.append(l_non_selected_representatives[0])
            # Update set with new elements that are not totally included by another
            set_done.update(eq_set)
        log.debug("l_representatives after: ", l_representatives)
        log.debug("set_done: ", set_done)
        set_done.difference_update(l_representatives)
        # log.debug("set_done.difference_update(l_representatives): ", set_done.difference_update(l_representatives))
        # log.debug("adjacency_matrix_with_equivalent: ", adjacency_matrix_with_equivalent)
        adj = adjacency_matrix_with_equivalent[l_representatives, :][:, l_representatives]
        log.debug("adj:\n", adj)
        l_selected_closures = [x for i, x in enumerate(self._elementary_cls) if i in l_representatives]
        log.debug("l_selected_closures: ", l_selected_closures)
        # if len(adj) != len(l_representatives):
            # log.debug("PROBLEM")
        self._clusters = l_selected_closures
        return self._clusters

    @property
    def l_selected_sets(self):
        """ Returns the final list of sets of the pseudohierarchy

        :returns: A matrix representing the relationship between the sets:
        adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 0 means i is a child of j
        adjacency_matrix_with_equivalent[i,j] == 1 and adjacency_matrix_with_equivalent[i,j] == 1 means i and j have no relationship
        :rtype: [numpy.ndarray]
        """
        if self._l_selected_sets is None:
            self.hierarchy()
        return self._l_selected_sets
