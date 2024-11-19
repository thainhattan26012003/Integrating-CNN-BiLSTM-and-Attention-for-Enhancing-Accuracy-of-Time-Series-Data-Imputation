import os
import pandas as pd
from select_model_run import *
from model_name import *
from metrics import *
from utils import *

foler_path = "P:\FA24\Paper_code\GWM_NOAA-202001-202004"

for file in os.listdir(foler_path):
    if file.endswith('.xlsx'):
        file_path = os.path.join(foler_path, file)

df = pd.read_excel("P:\FA24\Paper_code\GWM_NOAA-202001-202004\DATA__AINOSHIMA.xlsx")
df = df.iloc[1:, 1:2]
df = df.apply(pd.to_numeric, errors='coerce')


target_col = 'Observation'
r = 2 # r for checking the affect of missing values on training process, set higher if large data set
# methods = ['combine', 'BiLSTM', 'CNN', 'Attention', 'CNN_BiLSTM', 'BiLSTM_Attention', 'CNN_Attention']
methods = ['BiLSTM']
    
combine = []
CNN = []
BiLSTM = []
Attention = []
CNN_BiLSTM = []
BiLSTM_Attention = []
CNN_Attention = []

# for model_name in methods: 
#     if df[target_col].isna().any():
#         print("DataFrame had missing values.")
#         result_df = df.copy()
        
#         # Impute missing data by point first and then handle missing by gap by some methods later
#         imputed_points_df = imputed_points_missing_data(df=result_df, r=r)
        
#         # Apply DL or ML methods for imputation missing values in gap missing case
#         # Visualize the results between original dataset and after imputation all missing data for 2 cases
#         imputed_gap_df = run(model_name=model_name, df=imputed_points_df, target_col=target_col, r=r)
#         visualize_for_impute_real_missing_dataset(df=df, imputed_df=imputed_gap_df, target_col=target_col, name_of_dataset=str(file_path))
        
#     else:
#         print("DataFrame don't have missing values, creating !!!")
#         size_of_gap = input("Input number of missing values you want to create: ")
#         result_df, y_truth = create_continuous_missing_values(df, target_col, size_of_gap)
        
for model_name in methods:
    
    df_copy = df.copy()
        
    imputed_points_df = imputed_points_missing_data(df=df_copy, r=r)
    
    while True:
        nan_clusters = find_nan_clusters(imputed_points_df)
        
        if not nan_clusters:
            break
        
        start, end = nan_clusters[0]
        
        store_first_cluster = [
            value if (i == start or i == end) and np.isnan(value) else value 
            for i, value in enumerate(imputed_points_df)
            if not (np.isnan(value) and not (i == start or i == end))
        ]
        
        imputed_gap_df = run(model_name=model_name, df=imputed_points_df, target_col=target_col, r=r)
        visualize_for_impute_real_missing_dataset(df=df, imputed_df=imputed_gap_df, target_col=target_col, name_of_dataset=str(file_path))
        
        
        
    
            
        # print('combine:', average_performance_metrics(combine))
        # print('CNN:', average_performance_metrics(CNN))
        # print('BiLSTM:', average_performance_metrics(BiLSTM))
        # print('Attention:', average_performance_metrics(Attention))
        # print('CNN_BiLSTM:', average_performance_metrics(CNN_BiLSTM))
        # print('BiLSTM_Attention:', average_performance_metrics(BiLSTM_Attention))
        # print('CNN_Attention:', average_performance_metrics(CNN_Attention))`
