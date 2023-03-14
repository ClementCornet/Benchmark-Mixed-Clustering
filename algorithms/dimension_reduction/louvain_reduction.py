import numpy as np
import pandas as pd
from algorithms.utils.mixed_distances import huang_distance_matrix
from sknetwork.embedding import LouvainEmbedding, LouvainNE
import scipy as sp

import umap.umap_ as umap

def louvain_embedding(df, dim=None):
    t=len(df.columns)
    if dim is None:
        dim = len(df.columns)
    dist = huang_distance_matrix(df)
    kernel = pd.DataFrame(dist).apply(lambda x: np.exp(-x/t))
    louvain = LouvainNE(dim)
    #import streamlit as st
    #um = umap.UMAP(len(df.columns), metric='precomputed')
    #um.fit(dist)
    #graph = um.graph_
    #import streamlit as st
    #st.write(graph)
    #st.write(type(sp.sparse.csr_matrix(kernel)))
    #st.write(sp.sparse.csr_matrix(kernel))
    embedding = louvain.fit_transform(
        sp.sparse.csr_matrix(kernel))
    #import streamlit as st
    #st.write(embedding)
    return embedding