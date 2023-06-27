# Here we define the class in charge of stocking the space and the method for the pseudoclure
# We need to be more consistent about when the copy of a parameter is made
import numpy as np
from pretopologic.space.pretopological_space import PretopologicalSpace, Prenetwork


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


# pseudoclosure for a pretopologicalspace defined with families
def pseudoclosure_prenetwork(pre_space, initial_set):
    # We regroup all networks of the same prenetwork
    networks = [net for conj in pre_space.dnf for net in conj]
    unique_networks = np.unique(networks)
    used_networks = np.zeros(len(pre_space.network_index))
    used_networks[unique_networks] = 1

    for i, prenet in enumerate(pre_space.prenetworks):
        prenetwork_used = np.zeros(len(pre_space.network_index))
        prenetwork_used[pre_space.network_index == i] = 1
        threshold_indices = prenetwork_used*used_networks

        inf_set_t = np.copy(initial_set)
        strength_to_set = np.dot(prenet.network, inf_set_t)

        # We need to include the weights now !!
        # We will add couples of weight threshold
        for th in pre_space.threholds[threshold_indices.astype('bool')]:
            inf_set_t[strength_to_set >= th] = 1
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



if __name__ == "__main__":
    a = np.random.choice(np.arange(-50, 51), 2500).reshape(50, 50) / 10
    prenet = Prenetwork(a, [2])
    pre_space = PretopologicalSpace([prenet], [0])
    ans = best_ilp(pre_space, 5)
    print("done")