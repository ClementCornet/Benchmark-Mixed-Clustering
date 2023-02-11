import sys # Get sys.argv
import pandas as pd # Read CSV
from utils.gen_data import generate_data # Generate Data

# Imports Algos
import algorithms.kprototest


def get_algo(args):
    """
    Get the algorithm to profile from Command Line Arguments
    """
    if args[0] == "kprototest":
        return algorithms.kprototest.process
    return lambda x : x
    raise NotImplementedError("only kprototest for the moment")

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

@profile
def process_clustering(args):
    return get_algo(args)(get_data(args))

#@profile
def profiling():
    """
    Profiling  Computation Time and Memory Usage.
    Getting the algorithm name and the dataset to use from command line args (to combine with mprof)

    1st Command Line Arg - Algorithm to Profile
    2nd Argument - Name of CSV / "generated"
        if generated, following args are : n_clusters clust_std n_num n_cat cat_unique n_indiv

    """
    args = sys.argv[1:]
    _ = process_clustering(args)



if __name__ == '__main__':
    """
    This file is not supposed to be ran alone. Use mprof to profile functions.
        - mprof run --include-children --python python computation_cost.py <algorithm> <dataset>
        - mprof plot
        
    """
    profiling()
    
    #ll = []
    #for i in range(100_000_000):
    #    ll.append(i)