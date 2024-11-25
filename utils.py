import os
import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
from metrics import *

def to_df(array):   # convert to dataframe (optional)
    X = [i[:-1] for i in array]
    y = [i[-1] for i in array]
    transpose = [list(x) for x in zip(*X)]

    dataframe = pd.DataFrame({f'Column{i+1}': lst for i, lst in enumerate(transpose)})
    dataframe['Target'] = y
    return dataframe

def transform_to_multivariate(dataframe, gap_size): # convert univaritate data to multivariate
    lst_multi_var = []
    for i in range(len(dataframe) - gap_size):
        row = dataframe[i : i+gap_size+1]
        lst_multi_var.append(row)
    return np.array(lst_multi_var)

def missing_values(dataframe, column_name, num_missing_values):
    modified_df = dataframe.copy()
    
    if len(dataframe) > num_missing_values:
        random_index = random.randint(0, len(dataframe) - num_missing_values)
        modified_df.loc[random_index:random_index + num_missing_values - 1, column_name] = np.nan
    else:
        print("Error: The number of missing values requested exceeds the DataFrame's capacity.")
    return modified_df

def create_continous_missing_values(dataframe, column_name, num_missing_values, output_folder):
    try:
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        for i in range(1,11):
            
            output_path = os.path.join(output_folder, f'missing_{i}.csv')
            
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Skipping file creation.")
                continue
            
            modified_df = missing_values(dataframe, column_name, num_missing_values)
            
            modified_df.to_csv(output_path, index=False)
            
            print(f'{output_path} saved with continuous missing values.')

    except FileNotFoundError:
        print(f"Failed to find '{dataframe}'. Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")


def imputed_points_missing_data(df, max_continuous_missing_values):
    start = 0
    while start < len(df):
        if pd.isna(df.iloc[start, 0]):
            end = start
            while end < len(df) and pd.isna(df.iloc[end, 0]):
                end += 1
            
            # Check if size of missing <= 3 then apply interpolation
            if end - start <= max_continuous_missing_values:
                df.iloc[start:end] = df.iloc[start-1:end].interpolate(method='linear').iloc[1:]  # interpolate linearly

            start = end
        else:
            start += 1
            
    return df


# Split data after imputed missing by points into multiple clusters for handle each cluster
def find_nan_clusters(data):
    nan_positions = np.isnan(data)
    clusters = []
    start = None

    for idx, is_nan in enumerate(nan_positions):
        if is_nan:
            if start is None:
                start = idx
        elif start is not None:
            clusters.append((start, idx))
            start = None

    if start is not None:
        clusters.append((start, len(data) - 1))

    return clusters

def visualize_for_impute_real_missing_dataset(df, imputed_df, target_col, name_of_dataset):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    fig.suptitle(name_of_dataset, fontsize=16)
    
    axs[0].plot(df[target_col].values.tolist(), label='Initial Dataset', linestyle='-', color='black') 
    axs[0].set_title('Original Dataset')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Values')
    axs[0].legend()
    
    
    axs[1].plot(imputed_df[target_col].values.tolist(), label='Imputed Dataset', linestyle='-', color='orange')
    axs[1].set_title('Imputation Dataset')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Values')
    axs[1].legend()
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def visualize_for_impute_creation_missing_dataset(original_data, nan_index, size_of_gap, imputed_data, target_col, name_of_dataset, similarity_score):
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_data[target_col].values.tolist()[nan_index:nan_index+size_of_gap], label='Original Values')
    plt.plot(imputed_data, label='Imputed Values', linestyle='-')
    plt.title(f'Missing {name_of_dataset}, {round(similarity_score,4)}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()
