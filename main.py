import pandas as pd
from select_model_run import run
from model_name import select_model

from metrics import *
from utils import create_continuous_missing_values, visualize_for_impute_real_missing_dataset , check_cases_of_missing_data

df = pd.read_excel("P:\FA24\Paper_code\GWM_NOAA-202001-202004\DATA__AKITA.xlsx")
df = df.iloc[1:, 1:2]
df = df.apply(pd.to_numeric, errors='coerce')


target_col = 'Observation'
r = 2 # r for checking the affect of missing values on training process, set higher if large data set
# methods = ['combine', 'BiLSTM', 'CNN', 'Attention', 'CNN_BiLSTM', 'BiLSTM_Attention', 'CNN_Attention']
methods = ['CNN']
    
combine = []
CNN = []
BiLSTM = []
Attention = []
CNN_BiLSTM = []
BiLSTM_Attention = []
CNN_Attention = []

for model_name in methods: 
    if df[target_col].isna().any():
        print("DataFrame had missing values.")
        result_df = df
        filled_points_df = check_cases_of_missing_data(df=result_df, r=r)
        results = run(model_name=model_name, df=filled_points_df, target_col=target_col, r=r)
        visualize_for_impute_real_missing_dataset(df=filled_points_df, imputed_values=results, target_col=target_col)
        
    else:
        print("DataFrame don't have missing values, creating !!!")
        size_of_gap = input("Input number of missing values you want to create: ")
        result_df, y_truth = create_continuous_missing_values(df, target_col, size_of_gap)
    

               
        # print('combine:', average_performance_metrics(combine))
        # print('CNN:', average_performance_metrics(CNN))
        # print('BiLSTM:', average_performance_metrics(BiLSTM))
        # print('Attention:', average_performance_metrics(Attention))
        # print('CNN_BiLSTM:', average_performance_metrics(CNN_BiLSTM))
        # print('BiLSTM_Attention:', average_performance_metrics(BiLSTM_Attention))
        # print('CNN_Attention:', average_performance_metrics(CNN_Attention))`
