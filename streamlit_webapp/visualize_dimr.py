import streamlit as st
import plotly.express as px
from streamlit_webapp.utils import get_data
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding
from algorithms.dimension_reduction.umap_reduction import umap_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.louvain_reduction import louvain_embedding

def page():
    cols = st.columns([2,5])

    with cols[0].container():
        df = get_data()
    with cols[1].container():
        tab_famd, tab_lap, tab_um, tab_pm, tab_lv = st.tabs(['FAMD', 'Laplacian Eigenmaps', 'UMAP', 'PaCMAP', 'Louvain'])
        with tab_famd:
            famd_emb = famd_embedding(df, dim=3)
            st.plotly_chart(
                px.scatter_3d(famd_emb,0,1,2)
            )
            
        with tab_lap:
            lap = laplacian_embedding(df, dim=3)
            st.plotly_chart(
                px.scatter_3d(lap,0,1,2)
            )
        with tab_um:
            um = umap_embedding(df, n_components=3)
            st.plotly_chart(
                px.scatter_3d(um,0,1,2)
            )
        with tab_pm:
            pm = PaCMAP_embedding(df, n_components=3)
            st.plotly_chart(
                px.scatter_3d(pm,0,1,2)
            )
        with tab_lv:
            lv = louvain_embedding(df, 3)
            st.plotly_chart(
                px.scatter_3d(lv,0,1,2)
            )