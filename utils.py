import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

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

def create_continuous_missing_values(dataframe, column_name, num_missing_values): # randomly create missing data follow by gap_size(num_missing_values)
    modified_df = dataframe.copy()
    random_index = random.randint(0, len(dataframe) - num_missing_values - 1)
    y_truth = modified_df.loc[random_index:random_index + num_missing_values - 1, column_name].values.copy()
    modified_df.loc[random_index:random_index + num_missing_values - 1, column_name] = np.nan
    return modified_df, y_truth

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

    