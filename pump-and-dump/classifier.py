import datetime
import glob
import os
import re
from time import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import joblib

features_path = 'pump-and-dump/features/adapted_features/*csv'

one_month_start = "2021-11-01 00:00:00"
three_month_start = "2021-09-01 00:00:00"
six_month_start = "2021-06-01 00:00:00"
two_year_start = '2020-01-01 21:00:00'

end_date = "2021-12-01 23:00:00"

test_start_date = "2021-12-01"
test_end_date = "2022-01-01"


def create_training_testing_set(df, features, start_date_train, end_date_train, start_date_test, end_date_test):
    X_train = df[(df['date'] >= pd.to_datetime(start_date_train)) & (df['date'] <= pd.to_datetime(end_date_train))][features]
    Y_train = df[(df['date'] >= pd.to_datetime(start_date_train)) & (df['date'] <= pd.to_datetime(end_date_train))]['gt'].astype(int).values.ravel()
    
    X_test = df[(df['date'] >= pd.to_datetime(start_date_test)) & (df['date'] <= pd.to_datetime(end_date_test))][features]
    Y_test = df[(df['date'] >= pd.to_datetime(start_date_test)) & (df['date'] <= pd.to_datetime(end_date_test))]['gt'].astype(int).values.ravel()
    return X_train, Y_train, X_test, Y_test


def train_random_forest(X_train, Y_train, X_test, Y_test, file, crypto, type):
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
    clf.fit(X_train, Y_train)
    
    Y_predictions = clf.predict(X_test)
    #confusion_matrix = confusion_matrix(Y_test, Y_predictions)
    #print(confusion_matrix)
    
    results = {'data' : os.path.split(file)[1],
                'recall': [recall_score(Y_test, Y_predictions, average='macro')], 
                'precision' : [precision_score(Y_test, Y_predictions, average='macro')],
                'F1 score': [f1_score(Y_test, Y_predictions, average='macro')]}

    if not os.path.exists('pump-and-dump/results/random_forest/'+ crypto + '/' + type):
        os.makedirs('pump-and-dump/results/random_forest/'+ crypto + '/' + type)
            
    filename = 'pump-and-dump/results/random_forest/'+ crypto + '/' + type + '/' + os.path.splitext(os.path.basename(file))[0] + '.h5'
    print(filename)
    joblib.dump(clf, filename)

    return pd.DataFrame(results)
    

def train_logistic_regression(X_train, Y_train, X_test, Y_test, file, crypto, type):
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train, Y_train)
    
    Y_predictions = clf.predict(X_test)
    #confusion_matrix = confusion_matrix(Y_test, Y_predictions)
    #print(confusion_matrix)
    
    results = {'data' : os.path.split(file)[1],
                'recall': [recall_score(Y_test, Y_predictions, average='macro')], 
                'precision' : [precision_score(Y_test, Y_predictions, average='macro')],
                'F1 score': [f1_score(Y_test, Y_predictions, average='macro')]}

    if not os.path.exists('pump-and-dump/results/logistic_regression/'+ crypto + '/' + type):
        os.makedirs('pump-and-dump/results/logistic_regression/'+ crypto + '/' + type)

    filename = 'pump-and-dump/results/logistic_regression/'+ crypto + '/' + type + '/' + os.path.splitext(os.path.basename(file))[0] + '.h5'
    print(filename)
    joblib.dump(clf, filename)

    return pd.DataFrame(results)


def classifier(time_freq):

    features = [#'std_rush_order',
                #'avg_rush_order',
                'std_trades',
                'std_volume',
                'avg_volume',
                'std_price',
                'avg_price',
                'avg_price_max',
                'avg_price_min',
                'hour_sin',
                'hour_cos',
                'minute_sin',
                'minute_cos']

    cryptos = list(os.listdir('pump-and-dump/autoencoder_dataset/adapted'))
    #cryptos = list(set([((os.path.splitext(i)[0].split('_')[0])) for i in os.listdir('features/adapted_features')]))
    cryptos.append('all')

    # results for random forest
    one_month_results_rf = pd.DataFrame()
    three_month_results_rf = pd.DataFrame()
    six_month_results_rf = pd.DataFrame()
    two_year_results_rf = pd.DataFrame()
    
    # results for logistic regression
    one_month_results_lr = pd.DataFrame()
    three_month_results_lr = pd.DataFrame()
    six_month_results_lr = pd.DataFrame()
    two_year_results_lr = pd.DataFrame()

    for c in cryptos:
        crypto = os.path.splitext(os.path.basename(c))[0]
        print(crypto)
        
        for rolling_freq in [30]:
            file = 'pump-and-dump/features/adapted_features/' + crypto + "_" + time_freq + "_" + str(rolling_freq) +'.csv'
            print(file)
            computed_data = pd.read_csv(file, parse_dates=['date'])

            # one month training
            print('One month training')
            X_train_1m, Y_train_1m, X_test, Y_test = create_training_testing_set(computed_data, features, one_month_start, end_date, test_start_date, test_end_date)
            results = train_random_forest(X_train_1m, Y_train_1m, X_test, Y_test, file, crypto, type='one_month')
            one_month_results_rf = pd.concat([one_month_results_rf, results])
            results = train_logistic_regression(X_train_1m, Y_train_1m, X_test, Y_test, file, crypto, type='one_month')
            one_month_results_lr = pd.concat([one_month_results_lr, results])

            
            # three months training
            print('Three months training')
            X_train_3m, Y_train_3m, X_test, Y_test = create_training_testing_set(computed_data, features, three_month_start, end_date, test_start_date, test_end_date)
            results = train_random_forest(X_train_3m, Y_train_3m, X_test, Y_test, file, crypto, type='three_months')
            three_month_results_rf = pd.concat([three_month_results_rf, results])
            results = train_logistic_regression(X_train_3m, Y_train_3m, X_test, Y_test, file, crypto, type='three_months')
            three_month_results_lr = pd.concat([three_month_results_lr, results])

           
            # six months training
            print('Six months training')
            X_train_6m, Y_train_6m, X_test, Y_test = create_training_testing_set(computed_data, features, six_month_start, end_date, test_start_date, test_end_date)
            results = train_random_forest(X_train_6m, Y_train_6m, X_test, Y_test, file, crypto, type='six_months')
            six_month_results_rf = pd.concat([six_month_results_rf, results])
            results = train_logistic_regression(X_train_6m, Y_train_6m, X_test, Y_test, file, crypto, type='six_months')
            six_month_results_lr = pd.concat([six_month_results_lr, results])
            

            # two years training
            print('Two years training')
            X_train_2y, Y_train_2y, X_test, Y_test = create_training_testing_set(computed_data, features, two_year_start, end_date, test_start_date, test_end_date)
            results = train_random_forest(X_train_2y, Y_train_2y, X_test, Y_test, file, crypto, type="two_years")
            two_year_results_rf = pd.concat([two_year_results_rf, results])
            results = train_logistic_regression(X_train_6m, Y_train_6m, X_test, Y_test, file, crypto, type="two_years")
            two_year_results_lr = pd.concat([two_year_results_lr, results])


        one_month_results_rf.to_csv('pump-and-dump/results/one_month_rf.csv', index=False, float_format='%.3f')
        three_month_results_rf.to_csv('pump-and-dump/results/three_months_rf.csv', index=False, float_format='%.3f')
        six_month_results_rf.to_csv('pump-and-dump/results/six_months_rf.csv', index=False, float_format='%.3f')
        two_year_results_rf.to_csv('pump-and-dump/results/two_years_rf.csv', index=False, float_format='%.3f')

        one_month_results_lr.to_csv('pump-and-dump/results/one_month_lr.csv', index=False, float_format='%.3f')
        three_month_results_lr.to_csv('pump-and-dump/results/three_months_lr.csv', index=False, float_format='%.3f')
        six_month_results_lr.to_csv('pump-and-dump/results/six_months_lr.csv', index=False, float_format='%.3f')
        two_year_results_lr.to_csv('pump-and-dump/results/two_years_lr.csv', index=False, float_format='%.3f')


if __name__ == '__main__':
    start = datetime.datetime.now()
    classifier(time_freq='4H')
    print(datetime.datetime.now() - start)
