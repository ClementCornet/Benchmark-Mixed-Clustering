import pandas as pd
import json
import os
from subprocess import check_output
import numpy as np
from algorithms.utils.clustering_utils import elbow_method

#@profile
def process(df, **kwargs):
    """Process MixtComp
    
    Uses RMixtComp R package, interracting with Rscript Executable"""

    # Get the number of clusters to process
    k = elbow_method(df)

    # Passing the desired number of clusters k through a JSON file
    json_data = { "n_clusters" : str(k)}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    # Encode categorical variables to integers. They are still processed as categorical though.
    df2 = df.copy()
    df2.loc[:,df.dtypes == 'object']=df.loc[:,df.dtypes == 'object'].apply(
            lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

    # RMixtComp can't handle too large numbers
    for col in df2.columns:
        if df2[col].dtype != 'object':
            if np.max(df2[col]) > 100000:
                df2[col] = df2[col]/100

    # Write cleaned data to a CSV to pass it to R
    df2.columns = ["X"+str(i) for i in  range(df2.shape[1])]
    df2.to_csv('temp_data.csv', index=False)

    # MixtComp uses a different mixture distribution for each variable type
    d = dict(zip(df2.columns,df.dtypes))
    for k in d.keys():
        t = str(d[k])
        if t == 'object': # Categorical
            d[k] = 'Multinomial'
        elif t == 'int64': # Positive integers
            d[k] = 'Poisson'
        elif t == 'float64': # Continuous variable, possibly negative
            d[k] = 'Gaussian'
    
    # Write the Variables:Mixtures correspondance in a JSON file to pass it to R
    with open('model.json','w') as f:
        json.dump(d,f,indent=4)


    # Process Data through R
    try:
        check_output("Rscript algorithms/R_Scripts/mixtcomp.R", shell=True).decode()
    except:
        return [-1] * len(df)

    # Load Clustered Data written by R
    clusters = pd.read_csv('mixtcomp_temp.csv')

    clusters =  clusters['x']

    # Remove temp files created to interact with R
    os.remove('k.json')
    os.remove('model.json')
    os.remove('temp_data.csv')
    os.remove('mixtcomp_temp.csv')
    return clusters