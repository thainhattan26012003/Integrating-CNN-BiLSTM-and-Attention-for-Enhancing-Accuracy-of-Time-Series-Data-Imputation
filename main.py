import os
import pandas as pd
from select_model_run import *
from model_name import *
from metrics import *
from utils import *

foler_path = "GWM_NOAA-202001-202004"

for file in os.listdir(foler_path):
    if file.endswith('.xlsx'):
        file_path = os.path.join(foler_path, file)
        
    print("Processing file:", file_path)

    df = pd.read_excel(file_path)
    df = df.iloc[1:, 1:2]
    df = df.apply(pd.to_numeric, errors='coerce')

    target_col = 'Observation'
    max_continuous_missing_values = 2 # for handle points missing if it contain maximum 2 continuous values
    r = 2 # r for checking the affect of missing values on training process, set higher if large data set
    # methods = ['combine', 'BiLSTM', 'CNN', 'Attention', 'CNN_BiLSTM', 'BiLSTM_Attention', 'CNN_Attention']
    methods = ['CNN']

    for model_name in methods:
        
        if df[target_col].isna().any():
            df_copy = df.copy()
            while True:
                
                imputed_points_df = imputed_points_missing_data(df=df_copy, max_continuous_missing_values=max_continuous_missing_values)
                
                nan_clusters = find_nan_clusters(imputed_points_df.values)
                
                if not nan_clusters:
                    break

                start, end = nan_clusters[0]
                
                print(start, end)
                
                # get the data until the last position of the first nan cluster
                clusters_data = imputed_points_df.iloc[:end]
                
                # get the data after the first nan cluster with drop all the nan values of other clusters
                rest_data = imputed_points_df.iloc[end:].dropna().reset_index(drop=True)

                # combine 2 data to get the full data contain the first nan clusters
                first_nan_clusters_only = pd.concat([clusters_data, rest_data])

                # impute the first nan cluster
                results = run(model_name=model_name, df=first_nan_clusters_only, target_col=target_col, r=r)
                
                # replace the results values after imputation to original df, -> handle continous nan clusters
                imputed_points_df.iloc[start:end] = np.array(results).reshape(-1, 1)
            
        visualize_for_impute_real_missing_dataset(df=df, imputed_df=imputed_points_df, target_col=target_col, name_of_dataset=str(file_path))

        
    else:
        ...    

                
            # print('combine:', average_performance_metrics(combine))
            # print('CNN:', average_performance_metrics(CNN))
            # print('BiLSTM:', average_performance_metrics(BiLSTM))
            # print('Attention:', average_performance_metrics(Attention))
            # print('CNN_BiLSTM:', average_performance_metrics(CNN_BiLSTM))
            # print('BiLSTM_Attention:', average_performance_metrics(BiLSTM_Attention))
            # print('CNN_Attention:', average_performance_metrics(CNN_Attention))`
