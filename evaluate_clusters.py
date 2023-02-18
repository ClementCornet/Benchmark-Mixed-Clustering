from algorithms.measures.clustering_measures import internal_indices

from algorithms import famdkmeans

import pandas as pd


def evaluate_punctual(df, algo):
    clusters = algo(df)
    return internal_indices(df, clusters)


if __name__ == "__main__":
    df = pd.read_csv('data/penguins.csv').dropna()
    algo = famdkmeans.process
    #clusters = famdkmeans.process(df)

    print(
        evaluate_punctual(df, algo)
    )