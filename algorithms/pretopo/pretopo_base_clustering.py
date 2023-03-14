from sklearn.preprocessing import MinMaxScaler, StandardScaler
from algorithms.pretopo.pretopo_utils import prenetwork_closest, elementary_closures_shortest_degree, elementary_closures_shortest_degree_largeron
from algorithms.pretopo.pretopo_hierarchy import pseudohierarchy_filter_equivalents, pseudohierarchy_filter_threshold, pseudohierarchy_gravity

import numpy as np

#@profile
def pretopo_clustering(data):
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(data))
    clusters = pretopo_clusters.astype(str)
    return clusters

class Pretopocluster:

    

    def __init__(self):
        self.labels_ = np.array([])

    def fit(self, data):
        with recursion_depth(500000):
            self.labels_ = clustering(data)

    def fit_predict(self, data, distance_func=None, area_val=None):
        #import pandas as pd
        #data = pd.DataFrame(data)
        #for c in data.columns:
        #    if data[c].dtype != 'object':
        #        print(c)
        #        data[[c]] = StandardScaler().fit_transform(data[[c]])
        with recursion_depth(500000):
            self.labels_ = clustering(np.asarray(data),distance_func,area_val)
        return self.labels_

import sys
class recursion_depth:
    def __init__(self, limit):
        self.limit = limit
        self.default_limit = sys.getrecursionlimit()
    def __enter__(self):
        sys.setrecursionlimit(self.limit)
    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.default_limit)

def clustering(data, method="mine",distance_func=None,area_val=None):

    prenetwork = prenetwork_closest(data,distance_func)
    pre_space = PretopologicalSpace([prenetwork], [[0]])

    if method == "mine":
        print("mine")
        closures_list = elementary_closures_shortest_degree(pre_space, 20)
    else:
        print("largeron")
        closures_list = elementary_closures_shortest_degree_largeron(pre_space, 20)
    # print_closures_members(closures_list)
    # compare_closures(closures_list, closures_list_largeron)

    # GRAVITY PSEUDOHIERARCHY
    ph_gravity = pseudohierarchy_gravity(closures_list.copy())
    adj_pseudohierarchy_gravity = pseudohierarchy_filter_threshold(ph_gravity.copy(), 0.50)
    answer_gravity = pseudohierarchy_filter_equivalents(adj_pseudohierarchy_gravity.copy(), closures_list)
    filtered_adj_pseudohierarchy_gravity = answer_gravity[0]
    selected_closures_gravity = answer_gravity[1]
    # We select the closures that are at the end of the directed graph (Those that have only one neighbor, himself)
    final_selected_indices_gravity = np.argwhere(np.sum(filtered_adj_pseudohierarchy_gravity, axis=1) == 1).flatten()
    final_selected_closures_gravity = [x for i, x in enumerate(selected_closures_gravity) if i in final_selected_indices_gravity]

    # print("_______________SELECTED CLOSURES GRAVITY:")
    # print_closures_members(selected_closures_gravity)
    # print("_______________FINAL SELECTED CLOSURES GRAVITY:")
    # print_closures_members(final_selected_closures_gravity)

    import networkx as nx
    G=nx.Graph()
    G.add_edges_from(make_tree(adj_pseudohierarchy_gravity))
    pos = nx.spring_layout(G)
    pos = hierarchy_pos(G, -1, width=400)

    #import streamlit as st
    #st.write(pos)
    #st.write("(from pretopo base clustering l82)")

    labels = np.zeros(len(data))
    for i, closure in enumerate(final_selected_closures_gravity):
        labels += (i+1)*closure
        labels[labels > (i+1)] = (i+1)
    labels = labels - 1
    # !!!!!!!!!!! WE ARE ADDING THIS? MAYBE CAN CAUSE A PROBLEM IN THE OTHER EXAMPLE
    labels = labels.astype('int')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    cols = list(labels)
    cols.insert(0,-1)
    nx.draw(G, pos=pos, with_labels=False,node_size=10,node_color=cols)
    import streamlit as st
    #st.pyplot(fig)

    # PLOT
    # plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    # plot_clusters_2(data, final_selected_closures_gravity)

    return labels

def make_tree(adjacency_matrix):
    tree = []
    matrix_len = len(adjacency_matrix)
    for i in range(len(adjacency_matrix) - 1):
        j = i + 1
        done = True
        while j < matrix_len and done == True:
            if adjacency_matrix[i,j] == 1:
                tree.extend([(i, j)])
                done = False
            elif j == matrix_len - 1:
                tree.extend([(i, -1)])
                done = False
            else:
                j += 1
    tree.extend([(matrix_len - 1, -1)])
    return tree

def hierarchy_pos(G, root=None, width=50., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    import networkx as nx
    import random

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)




class PretopologicalSpace:

    def __init__(self, prenetworks, dnf):
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


# Here we define the network_family class. This will allow us to calculate faster the pseudoclosure for a family
# of networks with the same weights but different thresholds, or with the same threshold and connections, but the
# weights are augmented or diminished by a constant factor.
# network is a list of numpy squared arrays, they all have to have the same size
# threshold is an array of floats, this way we can define many appartenence functions for a same network
class Prenetwork:
    def __init__(self, network, thresholds, weights=[1]):
        self.network = network
        self.thresholds = thresholds
        self.weights = weights