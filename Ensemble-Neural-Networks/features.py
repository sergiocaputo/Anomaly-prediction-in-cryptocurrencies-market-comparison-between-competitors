from ctypes import resize
import glob
import datetime
import os
import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter

from sklearn.preprocessing import MinMaxScaler

path_original = 'autoencoder_dataset/original/*.csv'
path_adapted = 'autoencoder_dataset/adapted/*.csv'
features = ["symbol","Date","Open Price","High Price","Low Price","Close Price","Volume",
                "WAP","No.of Shares","No. of Trades","Total Turnover (Rs.)","Deliverable Quantity","% Deli. Qty to Traded Qty",
                "Spread High-Low","Spread Close-Open","Ch(t) Open Price","SMA Open Price","Ch(t) High Price","SMA High Price","Ch(t) Low Price",
                "SMA Low Price","Ch(t) Close Price","SMA Close Price","Label"]


def adapt_features():
    files = glob.glob(path_original)

    useless_features = ["Adj_Close","sentiment","VWAP","SMA_14","SMA_21","SMA_5","SMA_12","SMA_26","SMA_13","SMA_30","SMA_20","SMA_50","SMA_100","SMA_200",
    "EMA_14","EMA_21","EMA_5","EMA_12","EMA_26","EMA_13","EMA_30","EMA_20","EMA_50","EMA_100","EMA_200","RSI_14","RSI_21","RSI_5","RSI_12",
    "RSI_26","RSI_13","RSI_30","RSI_20","RSI_50","RSI_100","RSI_200","MACD_12_26_9","MACDH_12_26_9","MACDS_12_26_9","BBL_20","BBM_20",
    "BBU_20","MOM","STOCHF_14","STOCHF_3","STOCH_5","STOCH_3","CMO","DPO","UO","lag_1"]

    # wap media chiusura in 24 h
    # spread high low = high price - low price
    #spread close open
    #ch(t) open = roc_since_open = (close_price - self.open_price) / self.open_price *100 in 2 hours
    #sma sum (n close val)/n *100  n = 2 giorni


    if not os.path.exists('autoencoder_dataset/adapted'):
                os.makedirs('autoencoder_dataset/adapted')

    for f in files:
        print(f)
        df = pd.read_csv(f)

        df = df.drop(useless_features, axis=1)
        
        crypto = os.path.splitext(os.path.basename(f))[0]
        df['symbol'] = crypto
        
        df = df.rename(columns={'Open' : 'Open Price', 'High' : 'High Price', 'Low' : 'Low Price', 'Close' : 'Close Price',
                                'anomaly' : 'Label'})
                
        df['WAP'] = df['Close Price'].rolling(24).mean()
        
        df['SMA Open Price'] = df['Open Price'].rolling(30).mean()
        df['SMA High Price'] = df['High Price'].rolling(30).mean()
        df['SMA Low Price'] = df['Low Price'].rolling(30).mean()
        df['SMA Close Price'] = df['Close Price'].rolling(30).mean()

        for i in range(df.shape[0]):
            if i in range(0, 24):
                df.loc[i, 'WAP'] = df.loc[i, 'Close Price'] 
                        
            df.loc[i, 'Spread High-Low'] = df.loc[i, 'High Price'] - df.loc[i, 'Low Price']
            df.loc[i, 'Spread Close-Open'] = df.loc[i, 'Close Price'] - df.loc[i, 'Open Price']

            if i == 0:
                # no change in prices cause computed over 2 hours
                df.loc[i, 'Ch(t) Open Price'] = 0
                df.loc[i, 'Ch(t) High Price'] = 0
                df.loc[i, 'Ch(t) Low Price'] = 0 
                df.loc[i, 'Ch(t) Close Price'] = 0 
            else:
                df.loc[i, 'Ch(t) Open Price'] = (df.loc[i, 'Open Price'] - df.loc[i - 1, 'Open Price']) / df.loc[i - 1, 'Open Price']
                df.loc[i, 'Ch(t) High Price'] = (df.loc[i, 'High Price'] - df.loc[i - 1, 'High Price']) / df.loc[i - 1, 'High Price']
                df.loc[i, 'Ch(t) Low Price'] = (df.loc[i, 'Low Price'] - df.loc[i - 1, 'Low Price']) / df.loc[i - 1, 'Low Price']
                df.loc[i, 'Ch(t) Close Price'] =  (df.loc[i, 'Close Price'] - df.loc[i - 1, 'Close Price']) / df.loc[i - 1, 'Close Price']
            
            if i in range(0,30):
                df.loc[i, 'SMA Open Price'] = df.loc[i, 'Open Price']
                df.loc[i, 'SMA High Price'] = df.loc[i, 'High Price']
                df.loc[i, 'SMA Low Price'] = df.loc[i, 'Low Price']
                df.loc[i, 'SMA Close Price'] = df.loc[i, 'Close Price']           
        
        df['No.of Shares'] = 0
        df['No. of Trades'] = 0
        df['Total Turnover (Rs.)'] = 0
        df['Deliverable Quantity'] = 0
        df['% Deli. Qty to Traded Qty'] = 0

        df = df[features]
        df.to_csv('autoencoder_dataset/adapted/{}.csv'.format(crypto), index=False)
    

def preprocessing():
    files = glob.glob(path_adapted)
    features_to_normalize = ['WAP', 'SMA Open Price','SMA High Price','SMA Low Price','SMA Close Price','Spread High-Low','Spread Close-Open',
                             'Ch(t) Open Price','Ch(t) High Price','Ch(t) Low Price','Ch(t) Close Price']
    
    for f in files:
        print(f)
        crypto = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        
        scaler = MinMaxScaler()
        for col in df.columns:
            if col in features_to_normalize:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))

        for i in range(df.shape[0]):
            if df.loc[i, 'Label'] == 2:
                df.loc[i, 'Label'] = 1

        df.to_csv('autoencoder_dataset/adapted/{}.csv'.format(crypto), index=False)
    

if __name__ == '__main__':
    start = datetime.now()

    adapt_features()
    preprocessing()
    
    print(datetime.now() - start)
