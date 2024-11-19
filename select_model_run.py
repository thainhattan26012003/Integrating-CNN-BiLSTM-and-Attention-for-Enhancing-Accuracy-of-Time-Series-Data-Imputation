import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from model_name import *


from utils import *
from metrics import evaluate

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.0f}'.format)

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)


def one_direction(model_name, data, test_data, size_of_gap): 
    _df = to_df(data)

    # train process
    X_train = np.array(_df.iloc[:, :-1])
    y_train = np.array(_df.iloc[:, -1])
    
    model_select = select_model(model_name, X_train)

    model = model_select.fit(X_train, y_train, epochs=500, batch_size=256, callbacks=[early_stopping], validation_split=0.2)

    # evaluate process 
    test_data = np.concatenate(test_data).ravel()
    results = []
    for i in range(int(len(test_data)//2)):
        record = test_data[i:i+size_of_gap+1]
        
        # record is sub array, so if it is changed then main array is changed
        record[size_of_gap] = model_select.predict(np.array(record[:size_of_gap]).reshape(1, -1))
        results.append(record[size_of_gap])
    return results

def run(model_name, df, target_col, r):
    data = df[target_col].values.tolist()
    
    size_of_gap = df[target_col].isna().sum()

    nan_index = None
    for j, value in enumerate(data):
        if value != value:  # Check if the value is NaN
            nan_index = j
            break

    begining_of_data_checking = data[:(r*size_of_gap)+1]
    ending_of_data_checking = data[::-1][:r*size_of_gap+1]
    df_miss = data[nan_index:nan_index + size_of_gap]

    data_missing_ending = data[:nan_index]
    data_missing_begining  = data[nan_index+size_of_gap:][::-1] # inverse for get data feature

    # define case
    if all(value in begining_of_data_checking for value in df_miss): # case 1: missing values belong to first r*size_of_gap part of data 
        print("DATA MISSING BELONG TO THE FIRST PART OF DATA")
        data_transformed = transform_to_multivariate(data_missing_begining, size_of_gap)
        data_test = df.values.tolist()[nan_index : nan_index + 2 * size_of_gap][::-1]
        result = one_direction(model_name = model_name,
                               data=data_transformed,
                               test_data = data_test,
                               size_of_gap=size_of_gap)
        
    elif all(value in ending_of_data_checking for value in df_miss): # case 2: missing values belong to last r*size_of_gap part of data 
        print("DATA MISSING BELONG TO THE LAST PART OF DATA")
        data_transformed = transform_to_multivariate(data_missing_ending, size_of_gap)
        data_test = df.values.tolist()[nan_index - size_of_gap : nan_index + size_of_gap]
        result = one_direction(model_name = model_name,
                               data=data_transformed,
                               test_data = data_test,
                               size_of_gap=size_of_gap)
    
    else: # case: between
        print("DATA MISSING BELONG TO THE BETWEEN PART OF DATA")
        ''' a: after 
            b: before
        '''
        Da = data[nan_index + size_of_gap:][::-1]
        Db = data[:nan_index]

        MDb = transform_to_multivariate(Da, size_of_gap)
        data_test_before = df.values.tolist()[nan_index - size_of_gap : nan_index + size_of_gap]
        b_result = one_direction(model_name=model_name,
                                 data=MDb,
                                 test_data=data_test_before,
                                 size_of_gap=size_of_gap)
        
        MDa = transform_to_multivariate(Db, size_of_gap)  
        data_test_after = df.values.tolist()[nan_index:nan_index + 2 * size_of_gap ][::-1]    
        a_result = one_direction(model_name=model_name,
                                 data=MDa,
                                 test_data=data_test_after,
                                 size_of_gap=size_of_gap)

        final_result = [(x + y)/2 for x,y in zip(a_result, b_result)]
        result = final_result
    
    return result