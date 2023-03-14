import streamlit as st # Streamlit
import glob # Get list of .csv in data folder
import pandas as pd # Pandas
from utils.gen_data import generate_data # Generate data

# TODO : add st.form, to get submit button
def get_data():
    """
    Streamlit Widget to get the desired dataset.
    Local : Select a file among all CSV files in the `/data/` folder
    Generate : Generate a dataset using `utils.gen_data.generate_data`
    Uploaded (TODO) : Let the user upload his own CSV

    """
    data_method = st.radio('Get Data', ['Local','Generate','Uploaded'])
    if data_method == 'Local':
        files = glob.glob("data/*.csv")
        filename = st.selectbox('Filename', files)
        df =  pd.read_csv(filename).dropna().iloc[:5000]
        #for col in df.columns:
        #    if df[col].nunique() <= 5:
        #        df[col] = df[col].astype(str)
        st.dataframe(df)
        for col in df.columns:
            if df[col].nunique() <= 3:
                df[col] = df[col].astype('object')
        st.write(df.dtypes)
        return df
    if data_method == 'Generate':
        n_indiv = st.slider('Number of Individuals',min_value=50,max_value=5000,step=1)
        n_clusters = st.slider('Number of Clusters',min_value=2,max_value=10,step=1)
        n_num = st.slider('Number of Numerical Features',min_value=1,max_value=200,step=1)
        n_cat = st.slider('Number of Categorical Features',min_value=1,max_value=200,step=1)
        cat_u = st.slider('Number of Categorical Unique Values',min_value=2,max_value=50,step=1)
        clust_std = st.slider('Clusters Deviation',min_value=0.01,max_value=0.3,step=0.001)
        return generate_data(n_clusters,clust_std,n_num,n_cat,cat_u,n_indiv)
    if data_method == 'Uploaded':
        st.write('TODO')
        up = st.file_uploader('Upload File')    
        st.session_state['truth'] = False
        if up:
            df = pd.read_csv(up).dropna()
            st.write("????????????????????????")
            df.dropna(inplace=True)
            st.dataframe(df)
            st.write(df.isna().sum().sum())
            for col in df.columns:
                if df[col].nunique() <= 3:
                    df[col] = df[col].astype('object')
            st.write(df.dtypes)
            return df
        else:
            return pd.DataFrame()