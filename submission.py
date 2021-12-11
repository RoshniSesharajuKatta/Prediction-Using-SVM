import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import math


## Project-Part1
def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    #pass ## Replace this line with your implementation
    initial_train_df = train_df[["max_temp", "max_dew", "max_humid", "dailly_cases"]]
    transpose_df = initial_train_df.T # 5 rows x 162 columns
    required_features = ["max_temp", "max_dew", "max_humid", "past_cases"]
    column_names_list = []
    #120 columns names list creation
    for i in range(len(required_features)):
        for j in range(1, 31):
            x = 31 - j
            column_names_list.append(required_features[i]+"-"+ str(x))
    #print(column_names_list)
    
    #complete feature matrix
    feature_matrix = pd.DataFrame(columns = column_names_list)
    #print(feature_matrix)
    
    for i in range(31, len(initial_train_df)+1):
        temp = []
        for j in range(i-30, i):
            temp.append(j-1)
        
        new_df = transpose_df[temp]
        #print(new_df) # 4 rows x 30 columns
    
#     temp_index = []
#     for k in range(len(temp)):
#         temp_index.append(k + 1)
#     print(temp_index)
    
        new_df.columns = new_df.columns[::-1]
        #print(new_df) 
    
        new_df = new_df.stack().to_frame().T
        new_df.columns = ['{}-{}'.format(*c) for c in new_df.columns]
        new = new_df.set_axis(column_names_list[0::], axis = 1, inplace = False)
        #print(new)
        feature_matrix = feature_matrix.append(new, ignore_index = True)
        #print(feature_matrix)
    #print(feature_matrix)
    
    #selecting a subset of the complete feature matrix beased on the past_weather_interval and past_cases_interval values
    index_list = []
    col_list = []
    for m in feature_matrix:
        split_list = m.split("-")
        #print(split_list)  #['max_temp', '30']
        if split_list[0] == 'max_temp' or split_list[0] == 'max_dew' or split_list[0] == 'max_humid':
            if int(split_list[1]) <= past_weather_interval:
                #index_val = feature_matrix.columns.get_loc(m)
                #index_list.append(index_val)
                #appending the current column name
                col_list.append(m)
                #print(index_val)
                #print(index_list)
        if split_list[0] == 'past_cases':
            if int(split_list[1]) <= past_cases_interval:
                #index_val = feature_matrix.columns.get_loc(m)
                #index_list.append(index_val)
                #appending the current column name
                col_list.append(m)
                #print(index_list)
    #print(index_list)
    subset_of_feature_matrix = feature_matrix[col_list]
    #print(subset_of_feature_matrix)
    #x_train = subset_of_feature_matrix
    
    ###test feature processing###
    test_feature_df = test_feature.to_frame()
    test_feature_df = test_feature_df.T
    #print(test_feature_df.T)
    index_list_tf = []
    col_list_tf = []
    #selecting a subset of the test_feature_df
    for n in test_feature_df:
        #print(n)
        split_list_tf = n.split("-")
        #print(split_list)  #['max_temp', '30']
        if split_list_tf[0] == 'max_temp' or split_list_tf[0] == 'max_dew' or split_list_tf[0] == 'max_humid':
            if int(split_list_tf[1]) <= past_weather_interval:                
                #appending the current column name
                col_list_tf.append(n)
                
        if split_list_tf[0] == 'dailly_cases':
            if int(split_list_tf[1]) <= past_cases_interval:                
                #appending the current column name
                col_list_tf.append(n)
                
    #print(index_list)
    subset_of_test_matrix = test_feature_df[col_list_tf]
    #print(subset_of_test_matrix)
    
    ### training the model and prediction ###
    x_train = subset_of_feature_matrix
    #x_train
    y_train = train_labels_df['dailly_cases']
    y_train = y_train[30:]
    #print(y_train)
    
    x_test = subset_of_test_matrix
    #x_train 
    
    #fitting the svm model
    svm_model.fit(x_train, y_train)
    #predicting for x_test
    prediction = svm_model.predict(x_test)
    #converting the prediction to the nearest integer
    prediction = math.floor(prediction)
    return prediction



## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    #pass ## Replace this line with your implementation
    train_df = train_df[["max_temp", "avg_temp", "min_temp", "max_dew", "avg_dew", "min_dew", "max_humid", "avg_humid", "min_humid", "max_wind_speed", "avg_wind_speed", "min_wind_speed", "max_pressure", "avg_pressure", "min_pressure", "precipitation", "dailly_cases"]]
    transpose_df2 = train_df.T # 5 rows x 162 columns
    required_features2 = ["max_temp", "avg_temp", "min_temp", "max_dew", "avg_dew", "min_dew", "max_humid", "avg_humid", "min_humid", "max_wind_speed", "avg_wind_speed", "min_wind_speed", "max_pressure", "avg_pressure", "min_pressure", "precipitation", "dailly_cases"]
    column_names_list2 = []
    #510 columns names list creation
    for i in range(len(required_features2)):
        for j in range(1, 31):
            x = 31 - j
            column_names_list2.append(required_features2[i]+"-"+ str(x))
    #print(column_names_list)

    #complete feature matrix
    feature_matrix2 = pd.DataFrame(columns = column_names_list2)
    #print(feature_matrix)
    
    for i in range(31, len(train_df)+1):
        temp2 = []
        for j in range(i-30, i):
            temp2.append(j-1)

        new_df2 = transpose_df2[temp2]
        #print(new_df)

    #     temp_index = []
    #     for k in range(len(temp)):
    #         temp_index.append(k + 1)
    #     print(temp_index)

        new_df2.columns = new_df2.columns[::-1]
        #print(new_df) 

        new_df2 = new_df2.stack().to_frame().T
        new_df2.columns = ['{}-{}'.format(*c) for c in new_df2.columns]
        new2 = new_df2.set_axis(column_names_list2[0::], axis = 1, inplace = False)
        #print(new)
        feature_matrix2 = feature_matrix2.append(new2, ignore_index = True)
        #print(feature_matrix)
    #print(feature_matrix)
    
    #past_weather_interval = 20
    #past_cases_interval = 20
    #index_list = []
    col_list2 = []
    for m in feature_matrix2:
        split_list2 = m.split("-")
        #print(split_list)  #['max_temp', '30']
        if split_list2[0] == 'max_temp' or split_list2[0] == 'max_dew' or split_list2[0] == 'max_humid':
            if int(split_list2[1]) <= 20:
                #index_val = feature_matrix.columns.get_loc(m)
                #index_list.append(index_val)
                col_list2.append(m)
                #print(index_val)
                #print(index_list)
        if split_list2[0] == 'dailly_cases':
            if int(split_list2[1]) <= 20:
                #index_val = feature_matrix.columns.get_loc(m)
                #index_list.append(index_val)
                col_list2.append(m)
                #print(index_list)
    #print(index_list)
    

#     col_list = []
#     for m in feature_matrix:
#         split_list = m.split("-")
#         #print(split_list)  #['max_temp', '30']
#         if split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 30:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
#         if split_list[0] == 'max_temp' or split_list[0] == 'avg_temp' or split_list[0] == 'min_temp' : #or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
#         if split_list[0] == 'max_humid' or split_list[0] == 'avg_humid' or split_list[0] == 'min_humid': #or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 col_list.append(m)
#         if split_list[0] == 'max_wind_speed' or split_list[0] == 'avg_wind_speed' or split_list[0] == 'min_wind_speed': #or split_list[0] == 'max_pressure' or split_list[0] == 'min_pressure': #or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 col_list.append(m)
#         if split_list[0] == 'max_pressure' or split_list[0] == 'avg_pressure' or split_list[0] == 'min_pressure': #or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 col_list.append(m)
#         if split_list[0] == 'precipitation':
#             if int(split_list[1]) <= 20:
#                 col_list.append(m)
#         if split_list[0] == 'max_dew' or split_list[0] == 'avg_dew' or split_list[0] == 'min_dew': #or split_list[0] == 'max_humid': #or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 col_list.append(m)
            
#     subset_of_feature_matrix = feature_matrix[col_list]
#     #subset_of_feature_matrix
###############################

#     col_list = []
#     for m in feature_matrix:
#         split_list = m.split("-")
#         #print(split_list)  #['max_temp', '30']
#         if split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 30:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
#         if split_list[0] == 'max_temp' or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' : #or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 20:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
#         if split_list[0] == 'avg_temp' or split_list[0] == 'avg_dew' or split_list[0] == 'avg_humid' or split_list[0] == 'avg_wind_speed' or split_list[0] == 'avg_pressure' : #or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 30:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
                
#         if split_list[0] == 'min_temp' or split_list[0] == 'min_dew' or split_list[0] == 'min_humid' or split_list[0] == 'min_wind_speed' or split_list[0] == 'min_pressure' : #or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
#             if int(split_list[1]) <= 30:
#                 #index_val = feature_matrix.columns.get_loc(m)
#                 #index_list.append(index_val)
#                 col_list.append(m)
# #         if split_list[0] == 'precipitataion':# or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' : #or split_list[0] == 'max_dew' or split_list[0] == 'max_humid' or split_list[0] == 'max_wind_speed' or split_list[0] == 'max_pressure' or split_list[0] == 'dailly_cases':
# #             if int(split_list[1]) <= 10:
# #                 #index_val = feature_matrix.columns.get_loc(m)
# #                 #index_list.append(index_val)
# #                 col_list.append(m)
        
            
    subset_of_feature_matrix2 = feature_matrix2[col_list2]
    subset_of_feature_matrix2 = subset_of_feature_matrix2[60:]
    #subset_of_feature_matrix
    
    ###test feature processing###
    test_feature_df = test_feature.to_frame()
    test_feature_df = test_feature_df.T
    test_feature_df2 = test_feature_df[column_names_list2]
    #print(test_feature_df)
    test_feature_df2 = test_feature_df2[col_list2]
    #print(test_feature_df)
    
    ### training the model and prediction ###
    ## Set hyper-parameters for the SVM Model
    svm_model = SVR()
    svm_model.set_params(**{'kernel': 'poly', 'degree': 1, 'C': 10000,
                        'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})

    
    x_train = subset_of_feature_matrix2
    #x_train
    y_train = train_labels_df['dailly_cases']
    y_train = y_train[90:]
    #print(y_train)
    
    x_test = test_feature_df2
    #x_train 
    
    #fitting the svm model
    svm_model.fit(x_train, y_train)
    #predicting for x_test
    prediction = svm_model.predict(x_test)
    #converting the prediction to the nearest integer
    prediction = math.floor(prediction)
    return prediction