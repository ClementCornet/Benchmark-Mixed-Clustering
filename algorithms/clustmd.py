import pandas as pd
import json
import os
from subprocess import check_output
import numpy as np
from algorithms.utils.clustering_utils import elbow_method

#@profile
def process(df, **kwargs):
    """Process ClustMD
    
    Uses ClustMD R package, interracting with Rscript Executable"""

    k = elbow_method(df)

    # Write categorical and numerical values to split CSV files, as clustMD R package needs
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns

    df2 = df.copy()
    
    df2[categorical_columns]=df2[categorical_columns].apply(
            lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))
    #print(df[categorical_columns])
    df2[categorical_columns].to_csv('temp_cat.csv',index=False)

    df2[numerical_columns].to_csv('temp_continue.csv',index=False)

    # Pass the desired number of clusters to R through a json file
    json_data = { "n_clusters" : str(k)}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    # Calling Rscript Executable to process data
    try:
        check_output("Rscript algorithms/R_Scripts/clustmd.R", shell=True).decode()
    except:
        return [-1] * len(df)

    df_out = pd.read_csv('temp_clustered.csv') # R stored clusters vector in a file

    os.remove("temp_cat.csv")
    os.remove("temp_continue.csv")
    os.remove("temp_clustered.csv")
    os.remove("k.json")

    return df_out['x'].astype(str)