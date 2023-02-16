import pandas as pd
import os
from subprocess import check_output
from algorithms.utils.clustering_utils import elbow_method

import json


@profile
def process(df, **kwargs):
    """Process Modha-Spangler Algorithm
    
    Interacts with Rscript, gmsclust function of kamila R Package"""

    k = elbow_method(df)
    # Write numerical and categorical to split CSV files, to fit kamila R package's needs
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    df[categorical_columns].to_csv('temp_cat.csv',index=False)
    df[numerical_columns].to_csv('temp_continue.csv',index=False)

    # Pass the desired number of clusters to R through JSON
    json_data = { "n_clusters" : str(k)}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    # Process Data with R
    check_output("Rscript algorithms/R_Scripts/modha_spangler.R", shell=True).decode()

    # Load clustered data
    df_out = pd.read_csv('temp_clustered.csv')

    #df['cluster'] = df_out['cluster']

    # Remove temp files created to interact through R
    os.remove("temp_cat.csv")
    os.remove("temp_continue.csv")
    os.remove("temp_clustered.csv")
    os.remove("k.json")

    return df_out

    #return df