import os
import pandas as pd
from select_model_run import *
from model_name import *
from metrics import *
from utils import *


###############################################################################################
# 🚀 PLEASE PREPROCESS THE DATAFRAME ACCORDING TO THE STRUCTURE AND CONTENT OF YOUR DATA FILE.#
###############################################################################################


original_folder_path = "full_data_store"
create_missing_folder_path = 'missing_values_creation'

# for handle points missing if it contain maximum 2 continuous values
max_continuous_missing_values = 2 

# r for checking the affect of missing values on training process
r = 2 


# Dictionary to store the results to compute mean of 10 times of imputation create time series missing data
model_lst_combine = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_CNN = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_BiLSTM = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_Attention = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_CNN_BiLSTM = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_CNN_Attention = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}
model_lst_BiLSTM_Attention = {'sim': [], 'mae': [], 'rmse': [], 'fsd': [], 'r': [], 'nse': []}

# Choose model to apply for all dataset
model_name = input("Please entern your model you want to work with: ")

for file in os.listdir(original_folder_path):
    file_path = os.path.join(original_folder_path, file)
    
    # read xlsx/excel file, colect the first column
    if file.endswith('.xlsx'): 
        df = pd.read_excel(file_path)
    
    # read csv file, colect the first column
    elif file.endswith('.csv'):
        df = pd.read_csv(file_path)
        
    # List all the features for user select        
    print("List of features of your data: ", list(df.columns))
        
    target_col = input("Select your column feature you want to process: ")
            
    if df[target_col].isna().any():
        print('Dataset has already misssing values!')
        
        df = df[[target_col]].iloc[1:]
        df = df.apply(pd.to_numeric, errors='coerce')
        
        while True:

            imputed_points_df = imputed_points_missing_data(df=df, max_continuous_missing_values=max_continuous_missing_values)
            
            nan_clusters = find_nan_clusters(imputed_points_df.values)

            # if no any nan values, break
            if not nan_clusters:
                break

            start, end = nan_clusters[0]
            
            # get the data until the last position of the first nan cluster
            clusters_data = imputed_points_df.iloc[:end]
            
            # get the data after the first nan cluster with drop all the nan values of other clusters
            rest_data = imputed_points_df.iloc[end:].dropna().reset_index(drop=True)

            # combine 2 data to get the full data contain the first nan clusters
            first_nan_clusters_only = pd.concat([clusters_data, rest_data])

            # impute the first nan cluster
            results, nan_index, size_of_gap = run(model_name=model_name, df=first_nan_clusters_only, target_col=target_col, r=r)
            
            # replace the results values after imputation to original df, -> handle continous nan clusters
            imputed_points_df.iloc[start:end] = np.array(results).reshape(-1, 1)

        visualize_for_impute_real_missing_dataset(df=df, imputed_df=imputed_points_df, target_col=target_col, name_of_dataset=str(file_path))

    
    else:
        print('Dataset has not already misssing values, create for testing accuracy!')
        
        num_missing_values = int(input("Please enter your continous missing values you want to create: "))
        
        # make missing file creation folder contain continous missing values
        create_continous_missing_values(dataframe=df, column_name=target_col, num_missing_values=num_missing_values, output_folder=create_missing_folder_path)
        
        for file in os.listdir(create_missing_folder_path):
            
            file_path = os.path.join(create_missing_folder_path, file)
            
            df_miss_values = pd.read_csv(file_path)
                
            results, nan_index, size_of_gap = run(model_name=model_name, df=df_miss_values, target_col=target_col, r=r)

            similarity_score, MAE_score, RMSE_score, FSD_score, NSE_score = calculate_metrics(results, model_name, nan_index, size_of_gap, model_lst_CNN)
            
            visualize_for_impute_creation_missing_dataset(df_miss_values[target_col].values.tolist()[nan_index:nan_index+size_of_gap],
                                                          nan_index=nan_index,
                                                          size_of_gap=size_of_gap,
                                                          imputed_data=results,
                                                          target_col=target_col,
                                                          name_of_dataset=str(file_path),
                                                          similarity_score=similarity_score)
                
            # print('combine:', average_performance_metrics(combine))
            # print('CNN:', average_performance_metrics(CNN))
            # print('BiLSTM:', average_performance_metrics(BiLSTM))
            # print('Attention:', average_performance_metrics(Attention))
            # print('CNN_BiLSTM:', average_performance_metrics(CNN_BiLSTM))
            # print('BiLSTM_Attention:', average_performance_metrics(BiLSTM_Attention))
            # print('CNN_Attention:', average_performance_metrics(CNN_Attention))`
