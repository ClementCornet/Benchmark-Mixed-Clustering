import streamlit as st # Streamlit
from streamlit_webapp.utils import get_data
import pandas as pd # Pandas
import plotly.express as px # Plotting
import pickle
import numpy as np

# Algorithms
from algorithms import denseclus,famdkmeans,hierar_gower,kamila,kproto, pretopomd, \
                       mixtcomp,modha_spangler, pretopo_FAMD, pretopo_laplacian, pretopo_UMAP, pretopo_PaCMAP, \
                       clustmd, pretopo_louvain

from algorithms.measures.clustering_measures import internal_indices
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import gower
from algorithms.dimension_reduction.famd_reduction import famd_embedding
from algorithms.dimension_reduction.laplacian_reduction import laplacian_embedding
from algorithms.dimension_reduction.pacmap_reduction import PaCMAP_embedding
from algorithms.dimension_reduction.umap_reduction import umap_embedding


def page_2():
    import numpy as np
    np.random.seed(1)
    cols = st.columns([2,5])
    with cols[0].container():
        df = get_data()
    with cols[1].container():
        clusters = {
            "ClustMD":clustmd.process(df),
            "DenseClus":denseclus.process(df),
            "Phillip & Ottaway":hierar_gower.process(df),
            "Kamila":kamila.process(df),
            "K-Prototypes":kproto.process(df),
            "MixtComp":mixtcomp.process(df),
            "Modha-Spangler":modha_spangler.process(df),
            "Pretopo-FAMD":pretopo_FAMD.process(df),
            "Pretopo-UMAP":pretopo_UMAP.process(df),
            "Pretopo-PaCMAP":pretopo_PaCMAP.process(df),
            "PretopoMD":pretopomd.process(df)
        }

        # print("saving labels to pickle")
        # with open('labels_on_features_and_ts_extracted_features.pickle', 'wb') as handle:
        #     pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df.to_csv('data.csv', index=False)

        labels_df = pd.DataFrame(clusters)
        labels_df.to_csv('labels.csv', index=False)

        # Open the pickle file for reading
        # with open('updated_labels.pkl', 'rb') as f:
        #     # Load the pickle dict
        #     clusters = pickle.load(f)

        print(clusters)

        famd_emb = famd_embedding(df)
        gmat = gower.gower_matrix(df)

        CH = {k:safe_CH(famd_emb, v) for k,v in clusters.items()}
        DB = {k:safe_DB(famd_emb, v) for k,v in clusters.items()}
        Si = {k:safe_Si(famd_emb, v) for k,v in clusters.items()}
        G_Si = {k:safe_Si(gmat, v, metric='precomputed') for k,v in clusters.items()}

        df = pd.DataFrame(columns = ['Calinski-Harabasz', 'Silhouette FAMD', 'Silouhette Gower',
                                     'Davies-Bouldin'])
        #df.columns = ['Calinski-Harabasz', 'Silhouette FAMD', 'Silouhette Gower', 'Davies-Bouldin']
        df['Calinski-Harabasz'] = pd.Series(CH.values())
        df['Silhouette FAMD'] = pd.Series(Si.values())
        df['Silouhette Gower'] = pd.Series(G_Si.values())
        df['Davies-Bouldin'] = pd.Series(DB.values())

        df.index = clusters.keys()

        df.to_csv('indexes.csv')

        st.info(df.to_latex())
        #st.dataframe(pd.DataFrame(index=['Calinski-Harabasz', 'Silhouette FAMD', 'Silouhette Gower', 'Davies-Bouldin'],
        #                          CH.values, ))

        print("saving indicators to pickle")
        with open('CH.pickle', 'wb') as handle:
            pickle.dump(CH, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('DB.pickle', 'wb') as handle:
            pickle.dump(DB, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Si.pickle', 'wb') as handle:
            pickle.dump(Si, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('G_Si.pickle', 'wb') as handle:
            pickle.dump(G_Si, handle, protocol=pickle.HIGHEST_PROTOCOL)



        t1, t2, t3, t4, t_ari_ami, t5 = st.tabs(
            ['FAMD Calinski-Harabasz', 'FAMD Davies-Bouldin', 'FAMD Silhouette', 'Gower Silhouette', 'ARI-AMI',
             'Clusters'])
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

        with t_ari_ami.container():
            n_algos = len(clusters.keys())
            n_algos=11
            #ari = [[0] * n_algos] * n_algos
            #ami = [[0] * n_algos] * n_algos
            import numpy as np
            iijj = np.zeros((11,11))
            ari = np.zeros((11,11))
            ami = np.zeros((11,11))
            for i in range(11):
                for j in range(11):
                    c1 = clusters[list(clusters.keys())[i]]
                    c2 = clusters[list(clusters.keys())[j]]
                    iijj[i][j] = 10*i+j
                    #st.write(f"{list(clusters.keys())[i]} {list(clusters.keys())[j]} {adjusted_rand_score(c1, c2)}")
                    ari[i][j] = adjusted_rand_score(c1, c2)
                    ami[i][j] = adjusted_mutual_info_score(c1, c2)
                    #printzzz
            #st.dataframe(pd.DataFrame(iijj))
            #st.write(iijj)
            import seaborn as sns
            sns.set(font_scale=0.7)
            #fig, ax = sns.heatmap(ari)
            #heat.x
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            fig.set_size_inches(4,4)
            sns.heatmap(pd.DataFrame(ari,columns=list(clusters.keys()),index=list(clusters.keys())),
                        annot=True,square=True,cbar=False,fmt=".2f",cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
            #st.dataframe(pd.DataFrame(ari,columns=list(clusters.keys()),index=list(clusters.keys())))
            #ax.set_title(data)
            #sns.set(font_scale=1)
            ax.set_title("Adjusted Rand Index")

            st.pyplot(fig)

            fig2,ax2 = plt.subplots()
            fig2.set_size_inches(4,4)
            sns.heatmap(pd.DataFrame(ami,columns=list(clusters.keys()),index=list(clusters.keys())),
                        annot=True,square=True,cbar=False,fmt=".2f",cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
            #st.dataframe(pd.DataFrame(ari,columns=list(clusters.keys()),index=list(clusters.keys())))
            #ax.set_title(data)
            #sns.set(font_scale=1)
            ax2.set_title("Adjusted Mutual Information")

            st.pyplot(fig2)

            #print(ari)

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