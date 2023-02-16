import numpy as np
import math
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler





# calculates an approximation of the distance between elements if they were evenly spaced
# uses the augmenting lines in the grid heuristic
def prenetwork_closest(points, distances=True, distance_func=None, area_val=None):
    ###### ORIGINAL VERSION : ONLY 2D
    x_dif = np.amax(points[:, 0]) - np.amin(points[:, 0])
    y_dif = np.amax(points[:, 1]) - np.amin(points[:, 1])
    area = x_dif*y_dif

    square_length = math.sqrt(area / len(points)) # sqrt of area for each point



    dm = distance_matrix(points, points)


    if distance_func is not None:
        dm = distance_func(points)
        square_length = np.mean(dm)
        print(square_length)


    closest = np.sum(np.ma.masked_less(dm, square_length/2).mask.astype('float'), axis=1)
    inverse = (closest-1)/closest
    real_points = len(points) - sum(inverse)
    radius = 2.5*square_length
    th = len(points)/(2*real_points)

    if distances:
        network = network_ball_distances(dm, points, radius)
    else:
        network = network_ball(dm, points, radius)
    return Prenetwork(network, [th])


def network_ball(dm, points, radius):
    adj = (dm < radius).astype('float')
    np.fill_diagonal(adj, 0)
    return adj

def network_ball_distances(dm, points, radius):
    adj = (dm > radius)
    dm[adj] = 0
    network = dm/radius
    network = 1 - network
    network[adj] = 0
    np.fill_diagonal(network, 0)
    return network


def elementary_closures_shortest_degree(pre_space, degree=2):
    dict_closures = {i: set() for i in range(degree, pre_space.size + 1)}
    # we make a network with all the
    total_network = np.zeros((pre_space.size, pre_space.size))
    for prenetwork in pre_space.prenetworks:
        total_network += abs(prenetwork.network)
    for i in np.arange(pre_space.size):
        initial_set = np.zeros(pre_space.size, dtype='float')
        initial_set[i] = 1
        for _ in range(degree-1):
            # we need to be careful if we try create a set bigger than the component size
            row = total_network[i, :].copy()
            row[initial_set.astype('bool')] = 0
            ma = np.ma.masked_equal(row, 0.0, copy=False)
            i_new = np.argmin(ma)
            initial_set[i_new] = 1
        dict_closures[degree].add(initial_set.tostring())
    print(len(dict_closures[degree]))
    # Maybe we start from zero, in case sets smaller than degree are created
    for i in np.arange(degree, pre_space.size):
        counter = 0
        level_closed = set()
        while dict_closures[i]:
            counter += 1
            if counter % 1000 == 0:
                print(i)
            initial_set = np.fromstring(dict_closures[i].pop(), dtype='float')
            pseudoclosure_set = pseudoclosure(pre_space, initial_set)
            length = int(sum(pseudoclosure_set))
            cls_str = pseudoclosure_set.tostring()
            if length > i:
                if cls_str not in dict_closures[length]:
                    dict_closures[length].add(cls_str)
            else:
                level_closed.add(cls_str)
        dict_closures[i] = level_closed.copy()
    return create_closures_list(dict_closures, degree)


def create_closures_list(dict_closures, degree):
    closures_list = list()
    noise_counter = 0
    for k, v in dict_closures.items():
        while v:
            temp = np.fromstring(v.pop(), dtype='float')

            # When working with clusters we eliminate the noise...
            if sum(temp) > degree + 3:
                closures_list.append(temp)
            else:
                noise_counter += 1

            # When comparing both closures algorithms we need all...
            # closures_list.append(temp)

    print("THE NOISE WAS: " + str(noise_counter))
    return closures_list


def elementary_closures_shortest_degree_largeron(pre_space, degree=2):
    # we make a network with all the
    total_network = np.zeros((pre_space.size, pre_space.size))
    for prenetwork in pre_space.prenetworks:
        total_network += abs(prenetwork.network)
    list_closures = list()
    for i in np.arange(pre_space.size):
        if i%200 == 0:
            print(i)
        initial_set = np.zeros(pre_space.size, dtype='float')
        initial_set[i] = 1
        for dist in range(degree-1):
            # we need to be careful if we try create a set bigger than the component size
            row = total_network[i, :].copy()
            row[initial_set.astype('bool')] = 0
            ma = np.ma.masked_equal(row, 0.0, copy=False)
            i_new = np.argmin(ma)
            initial_set[i_new] = 1
        list_closures.append(set_closure(pre_space, initial_set))
    return list_closures


def set_closure(pre_space, initial_set):
    set_pseudoclosure = pseudoclosure(pre_space, initial_set)
    if int(sum(initial_set)) == int(sum(set_pseudoclosure)):
        return set_pseudoclosure
    else:
        return set_closure(pre_space, set_pseudoclosure)

# pseudoclosure for a pretopologicalspace defined with families
def pseudoclosure(pre_space, initial_set):
    # We are calculating more than once the pseudoclosure, we need to stock it
    disj_list_pseudocl = list()
    for conj in pre_space.dnf:
        conj_list_pseudocl = list()
        for neigh_ind in conj:
            inf_set_t = np.copy(initial_set)
            neigh = pre_space.prenetworks[pre_space.network_index[neigh_ind]].network
            # Here is where need to adapt in order not to recalculate everything for the networks of a same family
            inf_set_t[np.dot(neigh, inf_set_t) >= pre_space.thresholds[neigh_ind]] = 1
            conj_list_pseudocl.append(inf_set_t)
        conj_pseudocl = np.logical_and.reduce(conj_list_pseudocl)
        disj_list_pseudocl.append(conj_pseudocl)
    return np.logical_or.reduce(disj_list_pseudocl).astype('float')

class Prenetwork:
    def __init__(self, network, thresholds, weights=[1]):
        self.network = network
        self.thresholds = thresholds
        self.weights = weights