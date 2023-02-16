import numpy as np
import random



def pseudohierarchy_gravity(elementary_cls_original):
    # We should try to vectorize this
    elementary_cls = np.array(elementary_cls_original.copy())
    adj = np.zeros((len(elementary_cls), len(elementary_cls)))
    for i, F in enumerate(elementary_cls):
        temporary_cls = elementary_cls[i:]

        intersections = np.dot(F, temporary_cls.T)
        sum_Gs = np.sum(temporary_cls, axis=1)

        # How much of G has F ?
        FhasG = intersections / sum_Gs

        # How much of F has G ?
        GhasF = intersections / sum(F)

        # How much of G is F ?
        FisG = sum(F) / sum_Gs

        # How much of F is G ?
        GisF = sum_Gs / sum(F)

        # An arrow from F to G weights the attraction that G has over F.
        adj[i, i:] = GisF*GhasF
        adj[i:, i] = FisG*FhasG
    return adj


def pseudohierarchy_filter_threshold(adj, threshold):
    adj[adj < threshold] = 0
    adj[adj >= threshold] = 1
    return adj


def pseudohierarchy_filter_equivalents(adj, closures):
    adj_equivalents = adj*adj.T
    list_equivalences = [list(np.argwhere(i == 1).flatten()) for i in adj_equivalents]
    list_representatives = list()
    set_done = set()
    for ind, eq_list in enumerate(list_equivalences):
        if ind not in set_done:
            list_representatives.append(random.choice(eq_list))
            set_done.update(eq_list)
    set_done.difference_update(list_representatives)
    adj = adj[list_representatives, :]
    adj = adj[:, list_representatives]
    selected_closures = [x for i, x in enumerate(closures) if i in list_representatives]
    if len(adj) != len(list_representatives):
        print("PROBLEM")
    return [adj, selected_closures]