import os
import pandas as pd
from select_model_run import *
from model_name import *
from metrics import *
from utils import *


###############################################################################################
# 🚀 PLEASE PREPROCESS THE DATAFRAME ACCORDING TO THE STRUCTURE AND CONTENT OF YOUR DATA FILE.#
###############################################################################################


original_folder_path = "GWM_NOAA-202001-202004"
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

model_dict = {
    'Combine': model_lst_combine,
    'CNN': model_lst_CNN,
    'BiLSTM': model_lst_BiLSTM,
    'Attention': model_lst_Attention,
    'CNN_BiLSTM': model_lst_CNN_BiLSTM,
    'CNN_Attention': model_lst_CNN_Attention,
    'BiLSTM_Attention': model_lst_BiLSTM_Attention
}

# Models to apply for all dataset
models_to_run = ['combine', 'CNN', 'BiLSTM', 'Attention', 'CNN_BiLSTM', 'CNN_Attention', 'BiLSTM_Attention']

for file in os.listdir(original_folder_path):
    file_path = os.path.join(original_folder_path, file)
    
    # Read file into DataFrame
    if file.endswith('.xlsx'): 
        df = pd.read_excel(file_path)
    elif file.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        continue  # Skip unsupported file types

    # List features and let user choose
    print("List of features of your data: ", list(df.columns))
    target_col = input("Select your column feature you want to process: ")
            
    if df[target_col].isna().any():
        print(f"Dataset '{file}' has missing values!")
        
        df = df[[target_col]].iloc[1:]
        df = df.apply(pd.to_numeric, errors='coerce')
        df_copy = df.copy()

        # Dictionary to store imputed results for each model
        model_results = {}

        for model_name in models_to_run:
            print(f"Running model: {model_name} on file: {file}")

            imputed_points_df = df_copy.copy()
            
            while True:
                imputed_points_df = imputed_points_missing_data(
                    df=imputed_points_df, 
                    max_continuous_missing_values=max_continuous_missing_values
                )
                
                nan_clusters = find_nan_clusters(imputed_points_df.values)
                if not nan_clusters:  # No more NaN clusters
                    break

                start, end = nan_clusters[0]
                clusters_data = imputed_points_df.iloc[:end]
                rest_data = imputed_points_df.iloc[end:].dropna().reset_index(drop=True)
                first_nan_clusters_only = pd.concat([clusters_data, rest_data])

                results, _, _ = run(
                    model_name=model_name, 
                    df=first_nan_clusters_only, 
                    target_col=target_col, 
                    r=r
                )
                
                # Update the DataFrame with imputed results
                imputed_points_df.iloc[start:end] = np.array(results).reshape(-1, 1)

            # Save results for visualization
            model_results[model_name] = imputed_points_df[target_col].values.tolist()

        # Visualize all models on one plot
        visualize_all_models_with_custom_layout(df, model_results, target_col, file)

    
    else:
        print('Dataset has not already misssing values, create for testing accuracy!')
        
        num_missing_values = int(input("Please enter your continous missing values you want to create: "))
        
        # make missing file creation folder contain continous missing values
        create_continous_missing_values(dataframe=df, column_name=target_col, num_missing_values=num_missing_values, output_folder=create_missing_folder_path)
        
        for file_creation in os.listdir(create_missing_folder_path):
            
            file_path_creation = os.path.join(create_missing_folder_path, file)
            
            df_miss_values = pd.read_csv(file_path_creation)
                
            results, nan_index, size_of_gap = run(model_name=model_name, df=df_miss_values, target_col=target_col, r=r)

            similarity_score, MAE_score, RMSE_score, FSD_score, NSE_score = calculate_metrics(file_path=file_path,
                                                                                              value_lst_after=results,
                                                                                              model_name=model_name,
                                                                                              nan_index=nan_index,
                                                                                              size_of_gap=size_of_gap,
                                                                                              model_dict=model_dict)
            
            visualize_for_impute_creation_missing_dataset(original_data=df,
                                                          nan_index=nan_index,
                                                          size_of_gap=size_of_gap,
                                                          imputed_data=results,
                                                          target_col=target_col,
                                                          name_of_dataset=str(file_path),
                                                          similarity_score=similarity_score)
        
        # Calculate metrics for each cases and models
        mean_metrics = calculate_mean_metrics(model_dict)

        # show the mean of metrics results
        for model_name, metrics in mean_metrics.items():
            print(f"\nMean metrics for {model_name}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
