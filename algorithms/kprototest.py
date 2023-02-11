from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
#from algos.utils import elbow_method


def process(df, **kwargs):
    
    k=6

    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    categorical_indexes = []

    # Scaling
    scaler = StandardScaler()
    for c in categorical_columns:
        categorical_indexes.append(df.columns.get_loc(c))
    if len(numerical_columns) == 0 or len(categorical_columns) == 0:
        return
    # create a copy of our data to be scaled
    df_scale = df.copy()
    # standard scale numerical features
    for c in numerical_columns:
        df_scale[c] = scaler.fit_transform(df[[c]])

    # Process Data
    kproto = KPrototypes(n_clusters=k)

    kproto.fit_predict(df_scale, categorical=categorical_indexes)

    #import streamlit as st
    #import pandas as pd
    #st.write("kproto")
    #st.write(pd.Series(kproto.labels_).value_counts())

    # return clusters as a list
    
    return  kproto.labels_