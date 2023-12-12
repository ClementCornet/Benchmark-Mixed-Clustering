import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_webapp.utils import get_data
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding
from algorithms.dimension_reduction.umap_reduction import umap_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.louvain_reduction import louvain_embedding

color_scale = [
        (0, "red"),
        (0.17, "orange"),
        (0.33, "yellow"),
        (0.5, "green"),
        (0.67, "blue"),
        (0.83, "indigo"),
        (1, "purple")
    ]
color_scale_if_outliers =[
        (0, "grey"),
        (0.17, "red"),
        (0.33, "yellow"),
        (0.5, "green"),
        (0.67, "blue"),
        (0.83, "indigo"),
        (1, "purple")
    ]


def page():
    cols = st.columns([2, 2])

    with cols[0].container():
        df = get_data()
        # Load cluster labels
        cluster_labels = pd.read_csv('labels.csv')

        # Make sure that the dataframes align
        if len(df) != len(cluster_labels):
            st.error("Data and cluster labels do not align. Please check your data.")
            st.stop()

        # Select clusterization
        cluster_options = cluster_labels.columns
        selected_cluster = st.selectbox("Select Clusterization", cluster_options)

        def plot_clusters(famd_emb, color):
            if -1 in color.values:

                tmp_color_scale = color_scale_if_outliers
            else:
                tmp_color_scale = color_scale
            st.plotly_chart(
                px.scatter_3d(famd_emb, 0, 1, 2, color=cluster_labels[selected_cluster],
                              color_continuous_scale=tmp_color_scale)
            )

    with cols[1].container():
        tab_famd, tab_lap, tab_um, tab_pm, tab_lv = st.tabs(
            ['FAMD', 'Laplacian Eigenmaps', 'UMAP', 'PaCMAP', 'Louvain'])

        with tab_famd:
            famd_emb = famd_embedding(df, dim=3)
            color = cluster_labels[selected_cluster]
            plot_clusters(famd_emb, color)

        with tab_lap:
            lap = laplacian_embedding(df, dim=3)
            color = cluster_labels[selected_cluster]
            plot_clusters(lap, color)

        with tab_um:
            um = umap_embedding(df, n_components=3)
            color = cluster_labels[selected_cluster]
            plot_clusters(um, color)

        with tab_pm:
            pm = PaCMAP_embedding(df, n_components=3)
            color = cluster_labels[selected_cluster]
            plot_clusters(pm, color)

        with tab_lv:
            lv = louvain_embedding(df, 3)
            color = cluster_labels[selected_cluster]
            plot_clusters(lv, color)

