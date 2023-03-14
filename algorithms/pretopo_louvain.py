from algorithms.pretopo.pretopo_base_clustering import pretopo_clustering
from algorithms.dimension_reduction.louvain_reduction import louvain_embedding


#@profile
def process(df, **kwargs):

    """
    Perform Pretopological Clustering on a mixed data set, in a tandem analysis with LOUVAIN
    
    """
    import streamlit as st
    st.write("louvain allez")
    embedding = louvain_embedding(df)

    clusters = pretopo_clustering(embedding)

    return clusters
