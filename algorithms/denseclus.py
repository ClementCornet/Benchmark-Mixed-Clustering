from denseclus import DenseClus

#@profile
def process(df, **kwargs):
    """Process HDBSCAN algorithm on dataset's UMAP coordinates.
    
    Based on Amazon-DenseClus python package"""
    
    clf = DenseClus(
        umap_combine_method="intersection_union_mapper",
    )
    clf.fit(df)

    return clf.score()