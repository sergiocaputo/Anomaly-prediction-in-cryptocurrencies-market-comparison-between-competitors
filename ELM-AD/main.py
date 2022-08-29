import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from datetime import datetime
from ELM import ELM
import numpy
import joblib

one_month_start = "2021-11-01 00:00:00"
three_month_start = "2021-09-01 00:00:00"
six_month_start = "2021-06-01 00:00:00"
two_year_start = '2020-01-01 21:00:00'

end_date = "2021-12-01 23:00:00"

test_start_date = "2021-12-01"
test_end_date = "2022-01-01"

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

def train_ELM(crypto, X_train, Y_train, X_test, Y_test, type, num_classes = 2, num_hidden_layers = 512, input_length = 20, complete_features = 'yes'):
    model = ELM(input_length, num_hidden_layers, num_classes)
    model.fit(X_train, Y_train, display_time=True)
    Y_predictions = model.predict(X_test)
    Y_predictions = define_prediction_class(Y_predictions)

    #confusionmatrix = confusion_matrix(Y_test, Y_predictions)
    #print(confusionmatrix)
    
    results = {'data' : crypto,
                'recall': [recall_score(Y_test, Y_predictions, average='macro')], 
                'precision' : [precision_score(Y_test, Y_predictions, average='macro')], 
                'F1 score': [f1_score(Y_test, Y_predictions, average='macro')]} 

    if complete_features == 'yes':
        if not os.path.exists('results/all_features/'+ crypto + '/' + type):
            os.makedirs('results/all_features/'+ crypto + '/' + type)
        filename = 'results/all_features/'+ crypto + '/' + type + '/' + 'ELM_' + crypto + '_' + type + '.h5'
    else:
        if not os.path.exists('results/reduced_features/'+ crypto + '/' + type):
            os.makedirs('results/reduced_features/'+ crypto + '/' + type)
        filename = 'results/reduced_features/'+ crypto + '/' + type + '/' + 'ELM_' + crypto + '_' + type + '.h5'

    print(filename)
    joblib.dump(model, filename)
    return pd.DataFrame(results)

def train_reduced_features_set():
    cryptos = list(os.listdir('autoencoder_dataset/adapted'))

    one_month_results_reduced = pd.DataFrame()
    three_month_results_reduced = pd.DataFrame()
    six_month_results_reduced = pd.DataFrame()
    two_year_results_reduced = pd.DataFrame()

    # models with features 'No.of Shares','No. of Trades','Total Turnover (Rs.)','Deliverable Quantity','% Deli. Qty to Traded Qty' removed
    # added 'Volume' feature

    reduced_features = ["Open Price","High Price","Low Price","Close Price","WAP","Volume",
                "Spread High-Low","Spread Close-Open","Ch(t) Open Price","SMA Open Price","Ch(t) High Price","SMA High Price","Ch(t) Low Price",
                "SMA Low Price","Ch(t) Close Price","SMA Close Price"]


    for c in cryptos:
        crypto = os.path.splitext(os.path.basename(c))[0]
        print(crypto)
        data = 'autoencoder_dataset/adapted/' + crypto +'.csv'

        computed_data = pd.read_csv(data, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, one_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_1m, Y_train_1m, X_test, Y_test, input_length=16, type='one_month', complete_features='no')
        one_month_results_reduced = pd.concat([one_month_results_reduced, results])
        
        print('Three months training')
        X_train_3m, Y_train_3m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, three_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_3m, Y_train_3m, X_test, Y_test, input_length=16, type='three_months', complete_features='no')
        three_month_results_reduced = pd.concat([three_month_results_reduced, results])

        print('Six months training')
        X_train_6m, Y_train_6m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, six_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_6m, Y_train_6m, X_test, Y_test, input_length=16, type='six_months', complete_features='no')
        six_month_results_reduced = pd.concat([six_month_results_reduced, results])

        print('Two years training')
        X_train_2y, Y_train_2y, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, two_year_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_2y, Y_train_2y, X_test, Y_test, input_length=16, type='two_years', complete_features='no')
        two_year_results_reduced = pd.concat([two_year_results_reduced, results])

    one_month_results_reduced.to_csv('results/reduced_features/one_month.csv', index=False, float_format='%.3f')
    three_month_results_reduced.to_csv('results/reduced_features/three_months.csv', index=False, float_format='%.3f')
    six_month_results_reduced.to_csv('results/reduced_features/six_months.csv', index=False, float_format='%.3f')
    two_year_results_reduced.to_csv('results/reduced_features/two_years.csv', index=False, float_format='%.3f')


def train_complete_features_set():
    cryptos = list(os.listdir('autoencoder_dataset/adapted'))
           
    one_month_results_complete = pd.DataFrame()
    three_month_results_complete = pd.DataFrame()
    six_month_results_complete = pd.DataFrame()
    two_year_results_complete = pd.DataFrame()

    complete_features = ["Open Price","High Price","Low Price","Close Price",
                "WAP","No.of Shares","No. of Trades","Total Turnover (Rs.)","Deliverable Quantity","% Deli. Qty to Traded Qty",
                "Spread High-Low","Spread Close-Open","Ch(t) Open Price","SMA Open Price","Ch(t) High Price","SMA High Price","Ch(t) Low Price",
                "SMA Low Price","Ch(t) Close Price","SMA Close Price"]


    for c in cryptos:
        crypto = os.path.splitext(os.path.basename(c))[0]
        print(crypto)
        data = 'autoencoder_dataset/adapted/' + crypto +'.csv'

        computed_data = pd.read_csv(data, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, one_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_1m, Y_train_1m, X_test, Y_test, type='one_month')
        one_month_results_complete = pd.concat([one_month_results_complete, results])
        
        print('Three months training')
        X_train_3m, Y_train_3m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, three_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_3m, Y_train_3m, X_test, Y_test, type='three_months')
        three_month_results_complete = pd.concat([three_month_results_complete, results])

        print('Six months training')
        X_train_6m, Y_train_6m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, six_month_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_6m, Y_train_6m, X_test, Y_test, type='six_months')
        six_month_results_complete = pd.concat([six_month_results_complete, results])

        print('Two years training')
        X_train_2y, Y_train_2y, X_test, Y_test = create_training_testing_set(computed_data, complete_features, two_year_start, end_date, test_start_date, test_end_date)
        results = train_ELM(crypto, X_train_2y, Y_train_2y, X_test, Y_test, type='two_years')
        two_year_results_complete = pd.concat([two_year_results_complete, results])

    one_month_results_complete.to_csv('results/all_features/one_month.csv', index=False, float_format='%.3f')
    three_month_results_complete.to_csv('results/all_features/three_months.csv', index=False, float_format='%.3f')
    six_month_results_complete.to_csv('results/all_features/six_months.csv', index=False, float_format='%.3f')
    two_year_results_complete.to_csv('results/all_features/two_years.csv', index=False, float_format='%.3f')


if __name__ == '__main__':
    start = datetime.now()
    
    train_complete_features_set()
    train_reduced_features_set()
    
    print(datetime.now() - start)