import streamlit as st # Streamlit
from streamlit_webapp.utils import get_data
import pandas as pd # Pandas
import plotly.express as px # Plotting

# Algorithms
from algorithms import denseclus,famdkmeans,hierar_gower,kamila,kproto,mixtcomp,modha_spangler, \
                pretopo_FAMD, pretopo_laplacian, pretopo_UMAP, pretopo_PaCMAP, clustmd, pretopo_louvain

from algorithms.measures.clustering_measures import internal_indices

from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import gower
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.umap_reduction import umap_embedding


#### TESTS #####
#from algorithms.zzztest import umap_hdbscan, new_laplacian_pretopo

def page():
    cols = st.columns([2,5])
    with cols[0].container():
        df = get_data()
    with cols[1].container():
        #st.write(df.dtypes)
        st.title("Compare Clustering Algorithms")
        clusters = {
            "DenseClus":denseclus.process(df),
            #"FAMD-KMeans":famdkmeans.process(df),
            "Phillip & Ottaway":hierar_gower.process(df),
            "Kamila":kamila.process(df),
            #"ClustMD":clustmd.process(df),
            "K-Prototypes":kproto.process(df),
            #"MixtComp":mixtcomp.process(df),
            #"Modha-Spangler":modha_spangler.process(df),
            "Pretopo-FAMD":pretopo_FAMD.process(df),
            "Pretopo-Laplacian":pretopo_laplacian.process(df),
            "Pretopo-UMAP":pretopo_UMAP.process(df),
            "Pretopo-PaCMAP":pretopo_PaCMAP.process(df),
            "Pretopo-Louvain":pretopo_louvain.process(df),
            #"Pretopo-PaCMAP-COPIER COLLER CA DOIT MARCHER JIHAZORGHAOZ":pretopo_PaCMAP.process2(df),
            #"UMAPHDBSCANAAAH":umap_hdbscan(df),
            #"NEWLAP":new_laplacian_pretopo(df)
        }
        
        with st.expander('Cluster Results'):
            for algo,res in clusters.items():
                st.write(algo)
                st.write(pd.Series(res).value_counts())

        indices = {k:internal_indices(df, v) for k,v in clusters.items()}

        for dimr in ['FAMD','Laplacian','UMAP','PaCMAP']:
            for score in ['CH', 'DB', 'Silhouette']:
                with st.expander(f"{dimr}-{score}"):
                    fig = px.bar(
                                x=clusters.keys(),
                                y=[indices[algo][dimr][score] for algo in clusters.keys()],
                                text_auto=True,
                                )
                    fig.update_xaxes(title_text="")
                    fig.update_yaxes(title_text="")
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    st.plotly_chart(fig)

        with st.expander('Gower-Silhouette'):
            fig = px.bar(
                            x=clusters.keys(),
                            y=[indices[algo]['Gower']['Silhouette'] for algo in clusters.keys()],
                            text_auto=True,
                        )
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig)


def page_2():
    cols = st.columns([2,5])
    with cols[0].container():
        df = get_data()
    with cols[1].container():
        clusters = {
            "DenseClus":denseclus.process(df),
            #"FAMD-KMeans":famdkmeans.process(df),
            "Phillip & Ottaway":hierar_gower.process(df),
            "Kamila":kamila.process(df),
            #"ClustMD":clustmd.process(df),
            "K-Prototypes":kproto.process(df),
            #"MixtComp":mixtcomp.process(df),
            #"Modha-Spangler":modha_spangler.process(df),
            "Pretopo-FAMD":pretopo_FAMD.process(df),
            "Pretopo-Laplacian":pretopo_laplacian.process(df),
            "Pretopo-UMAP":pretopo_UMAP.process(df),
            "Pretopo-PaCMAP":pretopo_PaCMAP.process(df),
            "Pretopo-Louvain":pretopo_louvain.process(df),
            #"Pretopo-PaCMAP-COPIER COLLER CA DOIT MARCHER JIHAZORGHAOZ":pretopo_PaCMAP.process2(df),
            #"UMAPHDBSCANAAAH":umap_hdbscan(df),
            #"NEWLAP":new_laplacian_pretopo(df)
        }

        famd_emb = famd_embedding(df)
        gmat = gower.gower_matrix(df)

        CH = {k:safe_CH(famd_emb, v) for k,v in clusters.items()}
        DB = {k:safe_DB(famd_emb, v) for k,v in clusters.items()}
        Si = {k:safe_Si(famd_emb, v) for k,v in clusters.items()}
        G_Si = {k:safe_Si(gmat, v, metric='precomputed') for k,v in clusters.items()}

        t1, t2, t3, t4, t5 = st.tabs(['FAMD Calinski-Harabasz', 'FAMD Davies-Bouldin', 'FAMD Silhouette', 'Gower Silhouette', 'Clusters'])
        with t1.container():
            fig = px.bar(x=CH.keys(),y=CH.values(),text_auto=True)
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig)
        with t2.container():
            fig = px.bar(x=DB.keys(),y=DB.values(),text_auto=True)
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig)
        with t3.container():
            fig = px.bar(x=Si.keys(),y=Si.values(),text_auto=True)
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig)
        with t4.container():
            fig = px.bar(x=G_Si.keys(),y=G_Si.values(),text_auto=True)
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig)
        with t5.container():
            for k,v in clusters.items():
                st.write(k)
                st.dataframe(pd.DataFrame(pd.Series(v).value_counts(),index=None).T)


def safe_CH(data, clusters):
    if pd.Series(clusters).nunique() <= 1:
        return 0
    return calinski_harabasz_score(data, clusters)

def safe_DB(data, clusters):
    if pd.Series(clusters).nunique() <= 1:
        return -1
    return davies_bouldin_score(data, clusters)

def safe_Si(data, clusters, **kwargs):
    if pd.Series(clusters).nunique() <= 1:
        return -1
    return silhouette_score(data, clusters, **kwargs)