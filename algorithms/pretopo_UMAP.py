from algorithms.pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import umap.umap_ as umap
from scipy.spatial.distance import cdist

from algorithms.pretopo.pretopo_base_clustering import pretopo_clustering
from algorithms.dimension_reduction.umap_reduction import umap_embedding

#@profile
def process(df, **kwargs):

    """
    Perform Pretopological Clustering on a mixed data set, in a tandem analysis with mixed UMAP.
    
    """

    embedding = umap_embedding(df)

    #import streamlit as st
    #st.dataframe(embedding)

    clusters = pretopo_clustering(embedding)

    return clusters
