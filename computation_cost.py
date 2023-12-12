import sys # Get sys.argv
import pandas as pd # Read CSV
from utils.gen_data import generate_data # Generate Data
from algorithms.utils.clustering_utils import elbow_method # Get the optimal number of clusters K
from algorithms.measures.clustering_measures import internal_indices # Evaluate Clusters
import json

# Imports Algos
import algorithms.kproto
import algorithms.pretopomd
import algorithms.denseclus
import algorithms.kamila
import algorithms.modha_spangler
import algorithms.mixtcomp
import algorithms.famdkmeans
import algorithms.hierar_gower
import algorithms.pretopo_UMAP
import algorithms.pretopo_PaCMAP
import algorithms.pretopo_laplacian
import algorithms.pretopo_FAMD
import algorithms.clustmd


def get_algo(args, **kwargs):
    """
    Get the algorithm to profile from Command Line Arguments

    """
    if args[0] == "denseclus":
        return algorithms.denseclus.process
    if args[0] == "kproto":
        return algorithms.kproto.process
    if args[0] == "pretopologic_euclidean_hamming":
        return algorithms.pretopologic_euclidean_hamming.process
    if args[0] == "kamila":
        return algorithms.kamila.process
    if args[0] == "modha_spangler":
        return algorithms.modha_spangler.process
    if args[0] == "mixtcomp":
        return algorithms.mixtcomp.process
    if args[0] == "famdkmeans":
        return algorithms.famdkmeans.process
    if args[0] == "hierar_gower":
        return algorithms.hierar_gower.process
    if args[0] == "pretopo_UMAP":
        return algorithms.pretopo_UMAP.process
    if args[0] == "pretopo_PaCMAP":
        return algorithms.pretopo_PaCMAP.process
    if args[0] == "pretopo_FAMD":
        return algorithms.pretopo_FAMD.process
    if args[0] == "pretopo_laplacian":
        return algorithms.pretopo_laplacian.process
    if args[0] == "clustmd":
        return algorithms.clustmd.process

    raise NotImplementedError("Wrong Algorithm Name")

#@profile 
def get_data(args):
    """
    Get dataset from command line, to benchmark computation cost of a given algorithm

    """
    if args[1] != "generated":
        return pd.read_csv(f"data/{args[1]}.csv").dropna() # i.e. 'penguins' for 'data/penguins.csv'
    gen_args = args[2:]
    return generate_data(
            n_clusters=int(gen_args[0]),
            clust_std=float(gen_args[1]),
            n_num=int(gen_args[2]),
            n_cat=int(gen_args[3]),
            cat_unique=int(gen_args[4]),
            n_indiv=int(gen_args[5])
        )

#@profile
def process_clustering(args):
    data = get_data(args)
    algo = get_algo(args)
    return data, algo(data)

#@profile
def profiling():
    """
    Profiling  Computation Time and Memory Usage.
    Getting the algorithm name and the dataset to use from command line args (to combine with mprof)

    1st Command Line Arg - Algorithm to Profile
    2nd Argument - Name of CSV / "generated"
        if generated, following args are : n_clusters clust_std n_num n_cat cat_unique n_indiv

    At the end, compute internal validation indices, and write their result to a file

    """
    args = sys.argv[1:]
    clusters = process_clustering(args) # ICI, METTRE EVALUATION + Better Plot
    #indices = internal_indices(clusters[0], clusters[1])
    #pd.DataFrame(indices).to_csv('indices.csv')



if __name__ == '__main__':
    """
    This file is not supposed to be ran alone. Use mprof to profile functions.
        - mprof run --include-children --python python computation_cost.py <algorithm> <dataset>
        - mprof plot
        
    """
    profiling()