from ctypes import resize
import glob
import datetime
import os
import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter

path_original = 'pump-and-dump/autoencoder_dataset/original/*.csv'
path_adapted = 'pump-and-dump/autoencoder_dataset/adapted/*.csv'


def std_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    #print(df_buy)
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def avg_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def std_trades_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].count()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    #print(len(results))

    return results


def std_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def avg_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def std_price_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def avg_price_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def avg_price_max(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def avg_price_min(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].min()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    #print(len(results))
    return results


def chunks_time(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    #the index contains time info
    return buy_volume.index

def majority_voting(df_buy_rolling):
    majority_labels = []

    for key, item in df_buy_rolling:
        gt_labels = []

        gt_labels.append(df_buy_rolling.get_group(key)['gt'].values)
        
        counts = Counter(gt_labels[0])
        most_frequent = counts.most_common(1)
        majority_labels.append(most_frequent[0][0])

    #print(majority_labels, len(majority_labels))
    
    return majority_labels


def build_features(file, coin, time_freq, rolling_freq, index):
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df['datetime'])
    df = df.reset_index().set_index('datetime')
    
    df_buy = df[df['side'] == 'buy']

    df_buy_grouped = df_buy.groupby(pd.Grouper(freq=time_freq)) #raggruppa per ogni 4 ore
    date = chunks_time(df_buy_grouped)
    
    results_df = pd.DataFrame(
        {'date': date,
         'pump_index': index,
         #'std_rush_order': std_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         #'avg_rush_order': avg_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         'std_trades': std_trades_feature(df_buy_grouped, rolling_freq).values,
         'std_volume': std_volume_feature(df_buy_grouped, rolling_freq).values,
         'avg_volume': avg_volume_feature(df_buy_grouped, rolling_freq).values,
         'std_price': std_price_feature(df_buy_grouped, rolling_freq).values,
         'avg_price': avg_price_feature(df_buy_grouped, rolling_freq).values,
         'avg_price_max': avg_price_max(df_buy_grouped, rolling_freq).values,
         'avg_price_min': avg_price_min(df_buy_grouped, rolling_freq).values,
         'hour_sin': np.sin(2 * np.pi * date.hour/23),
         'hour_cos': np.cos(2 * np.pi * date.hour/23),
         'minute_sin': np.sin(2 * np.pi * date.minute / 59),
         'minute_cos': np.cos(2 * np.pi * date.minute / 59),
         'gt': majority_voting(df_buy_grouped),
         })
    results_df['symbol'] = coin
    
    #for i in range(results_df.shape[0]):
    #    if results_df['gt'].iloc[i] == 2:
    #        results_df['gt'].iloc[i] = 1
    results_df.replace(np.inf, np.nan, inplace=True)

    return results_df.dropna()
    


def build_features_multi(time_freq, rolling_freq):

    files = glob.glob(path_adapted)

    all_results_df = pd.DataFrame()
    count = 0

    if not os.path.exists('pump-and-dump/features/adapted_features'):
                os.makedirs('pump-and-dump/features/adapted_features')
                
    for f in files:
        print(f, time_freq, rolling_freq)
        coin = os.path.splitext(os.path.basename(f))[0]

        results_df = build_features(f, coin, time_freq, rolling_freq, count)
        
        results_df.to_csv('pump-and-dump/features/adapted_features/{}_{}_{}.csv'.format(coin, time_freq, rolling_freq), index=False, float_format='%.3f')
        all_results_df = pd.concat([all_results_df, results_df])
        count += 1

    all_results_df.to_csv('pump-and-dump/features/adapted_features/all_{}_{}.csv'.format(time_freq, rolling_freq), index=False, float_format='%.3f')


def compute_features():
    # time step and time window
    # 4h cause curve shifting applied in costruction phase for dataset
    # window of elements considered (24 (1day), 30, 720 (1 month))

    #build_features_multi(time_freq='4H', rolling_freq=24)
    build_features_multi(time_freq='4H', rolling_freq=30)  
    #build_features_multi(time_freq='4H', rolling_freq=720)


def adapt_features():
    files = glob.glob(path_original)

    useless_features = ["Open","High","Low","Adj_Close","sentiment","VWAP","SMA_14","SMA_21","SMA_5","SMA_12","SMA_26","SMA_13","SMA_30","SMA_20","SMA_50","SMA_100","SMA_200",
    "EMA_14","EMA_21","EMA_5","EMA_12","EMA_26","EMA_13","EMA_30","EMA_20","EMA_50","EMA_100","EMA_200","RSI_14","RSI_21","RSI_5","RSI_12",
    "RSI_26","RSI_13","RSI_30","RSI_20","RSI_50","RSI_100","RSI_200","MACD_12_26_9","MACDH_12_26_9","MACDS_12_26_9","BBL_20","BBM_20",
    "BBU_20","MOM","STOCHF_14","STOCHF_3","STOCH_5","STOCH_3","CMO","DPO","UO","lag_1"]

    features = ["symbol","timestamp","datetime","side","price","amount","btc_volume","gt"]

    if not os.path.exists('pump-and-dump/autoencoder_dataset/adapted'):
                os.makedirs('pump-and-dump/autoencoder_dataset/adapted')

    for f in files:
        print(f)
        df = pd.read_csv(f)

        df = df.drop(useless_features, axis=1)
        
        crypto = os.path.splitext(os.path.basename(f))[0]
        df['symbol'] = crypto
        df = df.rename(columns={'Date' : 'datetime', 'Close' : 'price', 'Volume' : 'btc_volume', 'anomaly' : 'gt'})
        for i in range(df.shape[0]):
            df.loc[i, 'timestamp'] = datetime.timestamp(datetime.strptime( df.loc[i, 'datetime'], '%Y-%m-%d %H:%M:%S'))
            df.loc[i, 'amount'] = df.loc[i, 'btc_volume'] / df.loc[i, 'price']
        df['side'] = 'buy'
        
        df = df[features]
        df.to_csv('pump-and-dump/autoencoder_dataset/adapted/{}.csv'.format(crypto), index=False)
    


if __name__ == '__main__':
    start = datetime.now()

    adapt_features()
    compute_features()
    
    print(datetime.now() - start)
