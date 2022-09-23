import csv
from datetime import datetime
import os
from pyexpat import model
from unicodedata import name
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pydot_ng as pydot
#pydot.find_graphviz()
from keras.utils import plot_model
from keras.models import Model

from keras.layers import Dense
from keras.layers import concatenate

from sub_models import create_training_testing_set, define_prediction_class

one_month_start = "2021-11-01 00:00:00"
three_month_start = "2021-09-01 00:00:00"
six_month_start = "2021-06-01 00:00:00"
two_year_start = '2020-01-01 21:00:00'

end_date = "2021-12-01 23:00:00"

test_start_date = "2021-12-01"
test_end_date = "2022-01-01"


def load_all_models(models_path, n_models):
    all_models = list()
    for i in range(n_models):
        filename = models_path + str(i + 1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def define_stacked_model(models_path, members):
    ensemble_visible = []
    for i in range(len(members)):

        model = members[i]
        
        for layer in model.layers:
            layer.trainable = False     # set as true if the sub models need to be retrained
            layer._name = 'ensemble_' + str(i+1) + '_' + layer._name
                    
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    print(ensemble_visible)
    print()
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    print(ensemble_outputs)


    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file= models_path + 'model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy, model_path):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    history = model.fit(X, inputy, validation_split=0.33, epochs=50, batch_size=10)
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_path + "train-validation.png")

    model.save(model_path + 'ensamble.h5')

def predict_stacked_model(model, inputX):
    X = [inputX for _ in range(len(model.input))]
    return model.predict(X)

def train_ensamble(models_path, n_members, crypto, X_train, Y_train, X_test, Y_test):
    members = load_all_models(models_path, n_members)
    print('Loaded %d models' % len(members))
    stacked_model = define_stacked_model(models_path, members)
    fit_stacked_model(stacked_model, X_train, Y_train, models_path)
    Y_predictions = predict_stacked_model(stacked_model, X_test)
    Y_predictions = define_prediction_class(Y_predictions)
    #confusion_matrix = confusion_matrix(Y_test, Y_predictions)
    
    results = {'data' : crypto,
            'recall': [recall_score(Y_test, Y_predictions, average='macro')],       
            'precision' : [precision_score(Y_test, Y_predictions, average='macro')], 
            'F1 score': [f1_score(Y_test, Y_predictions, average='macro')]}  
    
    return pd.DataFrame(results)

def train_ensamble_reduced_features_set():
    cryptos = list(os.listdir('Ensemble-Neural-Networks/results/submodels/reduced_features'))
    n_members = 5

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
        data = 'Ensemble-Neural-Networks/autoencoder_dataset/adapted/' + crypto +'.csv'
        models_path = 'Ensemble-Neural-Networks/results/submodels/reduced_features/' + crypto

        computed_data = pd.read_csv(data, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, one_month_start, end_date, test_start_date, test_end_date)
        models_path_1m = models_path + '/one_month/'
        results = train_ensamble(models_path_1m, n_members, crypto, X_train_1m, Y_train_1m, X_test, Y_test)
        one_month_results_reduced = pd.concat([one_month_results_reduced, results])
        
        print('Three months training')
        X_train_3m, Y_train_3m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, three_month_start, end_date, test_start_date, test_end_date)
        models_path_3m = models_path + '/three_months/'
        results = train_ensamble(models_path_3m, n_members, crypto, X_train_3m, Y_train_3m, X_test, Y_test)
        three_month_results_reduced = pd.concat([three_month_results_reduced, results])

        print('Six months training')
        X_train_6m, Y_train_6m, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, six_month_start, end_date, test_start_date, test_end_date)
        models_path_6m = models_path + '/six_months/'
        results = train_ensamble(models_path_6m, n_members, crypto, X_train_6m, Y_train_6m, X_test, Y_test)
        six_month_results_reduced = pd.concat([six_month_results_reduced, results])

        print('Two years training')
        X_train_2y, Y_train_2y, X_test, Y_test = create_training_testing_set(computed_data, reduced_features, two_year_start, end_date, test_start_date, test_end_date)
        models_path_2y = models_path + '/two_years/'
        results = train_ensamble(models_path_2y, n_members, crypto, X_train_2y, Y_train_2y, X_test, Y_test)
        two_year_results_reduced = pd.concat([two_year_results_reduced, results])


    one_month_results_reduced.to_csv('Ensemble-Neural-Networks/results/submodels/reduced_features/one_month.csv', index=False, float_format='%.3f')
    three_month_results_reduced.to_csv('Ensemble-Neural-Networks/results/submodels/reduced_features/three_months.csv', index=False, float_format='%.3f')
    six_month_results_reduced.to_csv('Ensemble-Neural-Networks/results/submodels/reduced_features/six_months.csv', index=False, float_format='%.3f')
    two_year_results_reduced.to_csv('Ensemble-Neural-Networks/results/submodels/reduced_features/two_years.csv', index=False, float_format='%.3f')


def train_ensamble_complete_features_set():
    cryptos = list(os.listdir('Ensemble-Neural-Networks/results/submodels/all_features'))
    n_members = 5

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
        data = 'Ensemble-Neural-Networks/autoencoder_dataset/adapted/' + crypto +'.csv'
        models_path = 'Ensemble-Neural-Networks/results/submodels/all_features/' + crypto

        computed_data = pd.read_csv(data, parse_dates=['Date'])
        
        print('One month training')
        X_train_1m, Y_train_1m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, one_month_start, end_date, test_start_date, test_end_date)
        models_path_1m = models_path + '/one_month/'
        results = train_ensamble(models_path_1m, n_members, crypto, X_train_1m, Y_train_1m, X_test, Y_test)
        one_month_results_complete = pd.concat([one_month_results_complete, results])
        
        print('Three months training')
        X_train_3m, Y_train_3m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, three_month_start, end_date, test_start_date, test_end_date)
        models_path_3m = models_path + '/three_months/'
        results = train_ensamble(models_path_3m, n_members, crypto, X_train_3m, Y_train_3m, X_test, Y_test)
        three_month_results_complete = pd.concat([three_month_results_complete, results])

        print('Six months training')
        X_train_6m, Y_train_6m, X_test, Y_test = create_training_testing_set(computed_data, complete_features, six_month_start, end_date, test_start_date, test_end_date)
        models_path_6m = models_path + '/six_months/'
        results = train_ensamble(models_path_6m, n_members, crypto, X_train_6m, Y_train_6m, X_test, Y_test)
        six_month_results_complete = pd.concat([six_month_results_complete, results])

        print('Two years training')
        X_train_2y, Y_train_2y, X_test, Y_test = create_training_testing_set(computed_data, complete_features, two_year_start, end_date, test_start_date, test_end_date)
        models_path_2y = models_path + '/two_years/'
        results = train_ensamble(models_path_2y, n_members, crypto, X_train_2y, Y_train_2y, X_test, Y_test)
        two_year_results_complete = pd.concat([two_year_results_complete, results])


    one_month_results_complete.to_csv('Ensemble-Neural-Networks/results/submodels/all_features/one_month.csv', index=False, float_format='%.3f')
    three_month_results_complete.to_csv('Ensemble-Neural-Networks/results/submodels/all_features/three_months.csv', index=False, float_format='%.3f')
    six_month_results_complete.to_csv('Ensemble-Neural-Networks/results/submodels/all_features/six_months.csv', index=False, float_format='%.3f')
    two_year_results_complete.to_csv('Ensemble-Neural-Networks/results/submodels/all_features/two_years.csv', index=False, float_format='%.3f')


if __name__ == '__main__':
    start = datetime.now()
    train_ensamble_complete_features_set()
    train_ensamble_reduced_features_set()

    print(datetime.now() - start)