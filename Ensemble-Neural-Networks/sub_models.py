import csv
from datetime import datetime
import os
from tkinter.tix import Y_REGION
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import tensorflow as tf


one_month_start = "2021-11-01 00:00:00"
three_month_start = "2021-09-01 00:00:00"
six_month_start = "2021-06-01 00:00:00"
two_year_start = '2020-01-01 21:00:00'

end_date = "2021-12-01 23:00:00"

test_start_date = "2021-12-01"
test_end_date = "2022-01-01"


# fit model on dataset
def fit_model(trainX, trainy, complete_features, i):
    # define model
    model = Sequential()

    if complete_features == 'yes':
        model.add(Dense(20, input_dim=20, activation='relu', name = 'dense_input_ensemble_{}'.format(i) ))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
    else:
        model.add(Dense(16, input_dim=16, activation='relu', name = 'dense_input_ensemble_{}'.format(i) ))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))

    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=50)
    return model


def create_training_testing_set(df, features, start_date_train, end_date_train, start_date_test, end_date_test):
    X_train = df[(df['Date'] >= pd.to_datetime(start_date_train)) & (df['Date'] <= pd.to_datetime(end_date_train))][features].values.tolist()
    Y_train = df[(df['Date'] >= pd.to_datetime(start_date_train)) & (df['Date'] <= pd.to_datetime(end_date_train))]['Label'].astype(int).values.ravel()
    
    X_test = df[(df['Date'] >= pd.to_datetime(start_date_test)) & (df['Date'] <= pd.to_datetime(end_date_test))][features].values.tolist()
    Y_test = df[(df['Date'] >= pd.to_datetime(start_date_test)) & (df['Date'] <= pd.to_datetime(end_date_test))]['Label'].astype(int).values.ravel()
    
    return numpy.array(X_train), numpy.array(Y_train), numpy.array(X_test), numpy.array(Y_test)
    

def define_prediction_class(results):
    y = []
    for r in numpy.nditer(results):
        if r < float(0.5):
            y.append(0)
        else:
            y.append(1)
    return y


def train_sub_models(X_train, Y_train, crypto, type, complete_features='yes'):
    # train 5 sub models for each crypto
    n_members = 5
    #results = []
    for i in range(n_members):
        # fit model
        model = fit_model(X_train, Y_train, complete_features, i)

        # save model
        if complete_features == 'yes':
            if not os.path.exists('Ensemble-Neural-Networks/results/submodels/all_features/'+ crypto + '/' + type):
                os.makedirs('Ensemble-Neural-Networks/results/submodels/all_features/'+ crypto + '/' + type)
            filename = 'Ensemble-Neural-Networks/results/submodels/all_features/'+ crypto + '/' + type + '/' + str(i + 1) + '.h5'
        else:
            if not os.path.exists('Ensemble-Neural-Networks/results/submodels/reduced_features/'+ crypto + '/' + type):
                os.makedirs('Ensemble-Neural-Networks/results/submodels/reduced_features/'+ crypto + '/' + type)
            filename = 'Ensemble-Neural-Networks/results/submodels/reduced_features/'+ crypto + '/' + type + '/' + str(i + 1) + '.h5'

        model.save(filename)

        print('Saved %s' % filename)

        #return results


def train_complete_features():
    cryptos = list(os.listdir('Ensemble-Neural-Networks/autoencoder_dataset/adapted'))

    # models with features 'No.of Shares','No. of Trades','Total Turnover (Rs.)','Deliverable Quantity','% Deli. Qty to Traded Qty' = 0

    complete_features = ["Open Price","High Price","Low Price","Close Price",
                "WAP","No.of Shares","No. of Trades","Total Turnover (Rs.)","Deliverable Quantity","% Deli. Qty to Traded Qty",
                "Spread High-Low","Spread Close-Open","Ch(t) Open Price","SMA Open Price","Ch(t) High Price","SMA High Price","Ch(t) Low Price",
                "SMA Low Price","Ch(t) Close Price","SMA Close Price"]


    for c in cryptos:
        crypto = os.path.splitext(os.path.basename(c))[0]
        print(crypto)
        file = 'Ensemble-Neural-Networks/autoencoder_dataset/adapted/' + crypto +'.csv'

        computed_data = pd.read_csv(file, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, _ , _ = create_training_testing_set(computed_data, complete_features, one_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_1m, Y_train_1m, crypto, type="one_month")
        
        print('Three months training')
        X_train_3m, Y_train_3m, _ , _ = create_training_testing_set(computed_data, complete_features, three_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_3m, Y_train_3m, crypto, type="three_months")

        print('Six months training')
        X_train_6m, Y_train_6m, _ , _ = create_training_testing_set(computed_data, complete_features, six_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_6m, Y_train_6m, crypto, type="six_months")

        print('Two years training')
        X_train_2y, Y_train_2y, _ , _ = create_training_testing_set(computed_data, complete_features, two_year_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_2y, Y_train_2y, crypto, type="two_years")


def train_reduced_features_set():
    cryptos = list(os.listdir('Ensemble-Neural-Networks/autoencoder_dataset/adapted'))

    # models with features 'No.of Shares','No. of Trades','Total Turnover (Rs.)','Deliverable Quantity','% Deli. Qty to Traded Qty' removed
    # added 'Volume' feature

    reduced_features = ["Open Price","High Price","Low Price","Close Price","WAP","Volume",
                "Spread High-Low","Spread Close-Open","Ch(t) Open Price","SMA Open Price","Ch(t) High Price","SMA High Price","Ch(t) Low Price",
                "SMA Low Price","Ch(t) Close Price","SMA Close Price"]


    for c in cryptos:
        crypto = os.path.splitext(os.path.basename(c))[0]
        print(crypto)
        file = 'Ensemble-Neural-Networks/autoencoder_dataset/adapted/' + crypto +'.csv'

        computed_data = pd.read_csv(file, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, _ , _ = create_training_testing_set(computed_data, reduced_features, one_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_1m, Y_train_1m, crypto, type="one_month", complete_features='no')
        
        print('Three months training')
        X_train_3m, Y_train_3m, _ , _ = create_training_testing_set(computed_data, reduced_features, three_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_3m, Y_train_3m, crypto, type="three_months", complete_features='no')

        print('Six months training')
        X_train_6m, Y_train_6m, _ , _ = create_training_testing_set(computed_data, reduced_features, six_month_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_6m, Y_train_6m, crypto, type="six_months", complete_features='no')

        print('Two years training')
        X_train_2y, Y_train_2y, _ , _ = create_training_testing_set(computed_data, reduced_features, two_year_start, end_date, test_start_date, test_end_date)
        train_sub_models(X_train_2y, Y_train_2y, crypto, type="two_years", complete_features='no')

if __name__ == '__main__':
    start = datetime.now()
    
    train_complete_features()
    train_reduced_features_set()
    
    print(datetime.now() - start)