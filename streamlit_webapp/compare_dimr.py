import streamlit as st # Streamlit
from streamlit_webapp.utils import get_data
import pandas as pd # Pandas
import plotly.express as px

# Dimension Reductions
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding, lap_v2
from algorithms.dimension_reduction.umap_reduction import umap_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.louvain_reduction import louvain_embedding

from algorithms.measures.dataset_measures import hopkins_statistic, ivat
from algorithms.utils.clustering_utils import elbow_method

import numpy as np

#from pyclustertend import ivat
from sklearn.preprocessing import scale

from plotly.subplots import make_subplots

def page():
    """
    Compute the average Hopkins Statistic over 250 runs, for every available dimension reduction technique.

    """

    NUMBER_OF_RUNS = 2
    
    cols = st.columns([2,5])
    with cols[0].container():
    #with st.sidebar:
        df = get_data()
    with cols[1].container():
        if len(df)  == 0:
            return
        #st.write("allez")
        famd = [famd_embedding(df) for _ in range(NUMBER_OF_RUNS)]
        lap = [laplacian_embedding(df) for _ in range(NUMBER_OF_RUNS)]
        um = [umap_embedding(df) for _ in range(NUMBER_OF_RUNS)]
        pm = [PaCMAP_embedding(df) for _ in range(NUMBER_OF_RUNS)]
        lv = [louvain_embedding(df) for _ in range(NUMBER_OF_RUNS)]
        hop = {
            "FAMD" : np.mean([hopkins_statistic(f) for f in famd]),
            "Laplacian Eigenmaps" : np.mean([hopkins_statistic(l) for l in lap]),
            "UMAP" : np.mean([hopkins_statistic(u) for u in um]),
            "PaCMAP" : np.mean([hopkins_statistic(p) for p in pm]),
            "Louvain" : np.mean([hopkins_statistic(l) for l in lv]),
        }
        #st.write("dimr done")
        #st.write(hop)
        hop_fig = px.bar(x=hop.keys(), y=hop.values(),text_auto=True)
        hop_fig.update_xaxes(title_text="")
        hop_fig.update_yaxes(title_text="Hopkins Statistic")        
        ivat_list = [ivat(famd[0]), ivat(lap[0]), ivat(um[0]), ivat(pm[0]), ivat(lv[0])]
        
        # Scale ivats, to compensate plotly's auto scaling of pixels values
        iv_max = np.max(ivat_list)
        ivat_list = [iv*iv_max/np.max(iv) for iv in ivat_list] 
        ivat_fig = px.imshow(np.array(ivat_list), facet_col=0, binary_string=True,
                labels={'facet_col':'sigma'})
        for i in range(5):
            ivat_fig.layout.annotations[i]['text'] = f"{list(hop)[i]} : {round(float(hop[list(hop)[i]]),3)}"#['FAMD','Laplacian Eigenmaps', 'UMAP', 'PaCMAP'][i]
        ivat_fig.update_layout(margin=dict(t=20,b=0,l=0,r=0))
        ivat_fig.update_xaxes(showticklabels=False)
        ivat_fig.update_yaxes(showticklabels=False)

        hop_tab, ivat_tab = st.tabs(['Hopkins Statistic', 'iVAT'])
        with hop_tab:
            st.plotly_chart(hop_fig)
        with ivat_tab:
            st.plotly_chart(ivat_fig)