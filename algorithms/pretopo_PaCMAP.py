from algorithms.pretopo.pretopo_base_clustering import pretopo_clustering
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding


#@profile
def process(df, **kwargs):

    """
    Perform Pretopological Clustering on a mixed data set, in a tandem analysis with mixed PaCMAP.
    
    """

    df2 = df.copy()
    numerical = df2.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)

    embedding = PaCMAP_embedding(df2)
    clusters = pretopo_clustering(embedding)

    return clusters

from prince import FAMD
import pacmap
import pandas as pd
from algorithms.pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
def process2(df, **kwargs):
    famd = FAMD(n_components=len(df.columns)).fit_transform(df)
    fit = pacmap.PaCMAP(n_components=len(df.columns),apply_pca=False).fit_transform(famd)
    pm = pd.DataFrame(fit)
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(pm))

    # return clusters as a list
    clusters = pretopo_clusters.astype(str)

    #import streamlit as st
    #st.write("pacmapFAMD")
    #st.write(pd.Series(clusters).value_counts())

    from sklearn.metrics.cluster import calinski_harabasz_score
    import streamlit as st
    famd2 = FAMD(n_components=3).fit_transform(df)
    st.write(calinski_harabasz_score(famd2, clusters))
    return clusters