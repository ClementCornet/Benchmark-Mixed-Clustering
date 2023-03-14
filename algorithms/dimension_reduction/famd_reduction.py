import prince
import pandas as pd

#@profile
def famd_embedding(df, dim=None):
    """
    Get FAMD (Factorial Analysis of Mixed Data) embedding of a given mixed dataset.

    Parameters :
        df (pandas.DataFrame) : Mixed Dataset to process FAMD on
    
    Return:
        reduced (numpy.array) : FAMD coordinates of `df` in an Euclidean Space

    Note that the target space as the same number of dimensions as `df`, as the objective is to translate `df`
    in an Euclidean space more than reducing its size.
    
    """
    if dim is None:
        dim = len(df.columns)

    famd = prince.FAMD(n_components=dim) # Using the number of dimensions of the original data
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)
    #reduced = pd.DataFrame(reduced)
    #for i in range(dim):
    #    reduced.iloc[:,i] = reduced.iloc[:,i].apply(lambda x: x * famd.explained_inertia_[i])
    print(f"------------ {dim} dimensions | {100*famd.explained_inertia_.sum()}% inertia")

    return reduced