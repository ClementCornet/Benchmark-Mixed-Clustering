import prince

@profile
def famd_embedding(df):
    """
    Get FAMD (Factorial Analysis of Mixed Data) embedding of a given mixed dataset.

    Parameters :
        df (pandas.DataFrame) : Mixed Dataset to process FAMD on
    
    Return:
        reduced (numpy.array) : FAMD coordinates of `df` in an Euclidean Space

    Note that the target space as the same number of dimensions as `df`, as the objective is to translate `df`
    in an Euclidean space more than reducing its size.
    
    """

    famd = prince.FAMD(n_components=len(df.columns)) # Using the number of dimensions of the original data
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)
    return reduced