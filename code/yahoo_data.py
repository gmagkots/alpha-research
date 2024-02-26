import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def yahoo_fundamental():
    """
    Downloads from yahoo finance and saves a subset of the
    most recent fundamental data for the stock universe.

    :return: None
    """
    # desired feature names
    cols_income = ['EBIT', 'EBITDA', 'Net Income', 'Total Expenses', 'Basic EPS', 'Research And Development',
                   'Selling General And Administration', 'Gross Profit', 'Total Revenue']
    cols_balance = ['Ordinary Shares Number', 'Tangible Book Value', 'Invested Capital', 'Net Tangible Assets',
                    'Total Assets', 'Total Capitalization', 'Total Liabilities Net Minority Interest', 'Net PPE',
                    'Retained Earnings']
    cols_cash_flow = ['Capital Expenditure']

    # read the stock tickers and initialize output dataframe
    dfsym = pd.read_csv(os.path.join(OUT_DIR, 'tickers_industry.csv'))
    df = pd.DataFrame()

    # download and concatenate the annual statements for each stock
    for ticker in dfsym['ticker']:
        # print(ticker)
        hticker = yf.Ticker(ticker)
        dfi = hticker.income_stmt.T.rename_axis('date').reset_index()
        dfi = dfi[[col for col in dfi.columns if col in (['date'] + cols_income)]]
        dfb = hticker.balance_sheet.T.reset_index(drop=True)
        dfb = dfb[[col for col in dfb.columns if col in cols_balance]]
        dfc = hticker.cashflow.loc[cols_cash_flow].T.reset_index(drop=True)
        dfc = dfc[[col for col in dfc.columns if col in cols_cash_flow]]
        dft = pd.concat([dfi, dfb], axis=1)
        if dfc.empty:
            dft[cols_cash_flow] = np.nan
        else:
            dft = pd.concat([dft, dfc], axis=1)
        dft['ticker'] = ticker
        dft['currency'] = hticker.info['currency']
        df = pd.concat([df, dft])

    # save the output
    idcols = ['ticker', 'date']
    df = df[idcols + [col for col in df.columns if col not in idcols]]
    df.to_csv(os.path.join(OUT_DIR, 'data_raw_fundamental.csv'), index=False)


def yahoo_market():
    """
    Downloads from yahoo finance and saves a subset of the
    most recent market data for the stock universe.

    :return: None
    """
    # read the stock tickers and set the start/end dates
    dfsym = pd.read_csv(os.path.join(OUT_DIR, 'tickers_industry.csv'))
    start, end = '2010-01-01', (datetime.today() - BDay()).strftime('%Y-%m-%d')

    # read the historical stock prices, remove weekends
    # and tickers without any or bad data
    df = yf.download(dfsym['ticker'].tolist(), start=start, end=end, interval='1d', threads=False)['Adj Close']
    df = df.dropna(axis='columns', how='all').select_dtypes(exclude=['object']).reset_index()
    df = df[df['Date'].dt.dayofweek < 5]

    # convert wide to long format (RAM overfill issues for large stock universe)
    df = pd.melt(df, id_vars='Date', var_name='ticker', value_name='Adj Close')

    # save the stock prices
    df = df[['ticker', 'Date', 'Adj Close']].rename(columns={'Date': 'date', 'Adj Close': 'stock_price'})
    # df.to_csv(os.path.join(OUT_DIR, 'data_raw_prices.csv'), index=False)
    df.to_parquet(os.path.join(OUT_DIR, 'data_raw_prices.parquet'), index=False)


def yahoo_exchange_rates():
    """
    Get the exchange rates for the currencies of the stocks in the universe.
    Forward-fill the limited missing interim values.
    """
    # read the stock currencies and set the start/end dates
    dfc = pd.read_csv(os.path.join(OUT_DIR, 'data_raw_fundamental.csv'), usecols=['currency'])
    currs = dfc['currency'].dropna().unique().tolist()
    start, end = '2010-01-01', (datetime.today() - BDay()).strftime('%Y-%m-%d')

    # transform the list of currencies to exchange rate tickers
    currs = [curr + 'USD=X' for curr in currs if curr != 'USD']

    # read the historical fx rates and remove weekends and rates without any data
    df = yf.download(currs, start=start, end=end, interval='1d', threads=False)['Adj Close']
    df = df.dropna(axis='columns', how='all').select_dtypes(exclude=['object']).reset_index()
    df = df[df['Date'].dt.dayofweek < 5]

    # get fx rates from FRED and merge with the main data
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    fred_fx = ['DEXUSEU', 'DEXCHUS', 'DEXUSUK', 'DEXCAUS', 'DEXKOUS', 'DEXMXUS', 'DEXVZUS',
               'DEXINUS', 'DEXBZUS', 'DEXUSAL', 'DEXSZUS', 'DEXTHUS', 'DEXMAUS', 'DEXSFUS',
               'DEXTAUS', 'DEXHKUS', 'DEXSIUS', 'DEXSDUS', 'DEXUSNZ', 'DEXNOUS', 'DEXDNUS']
    fr = pdr.DataReader(fred_fx, data_source="fred", start=start, end=end)
    fr = fr.reset_index().rename(columns={'DATE': 'Date'})
    fr['Date'] = pd.to_datetime(fr['Date'])
    df = pd.merge(df, fr, on='Date', how='left')

    # backfill fx rates from yahoo finance with those from FRED
    # symbol mapping: DEXUSEU = EURUSD=X, DEXJPUS = 1/JPYUSD=X
    fx_map = {'CNYUSD=X': 'DEXCHUS', 'CADUSD=X': 'DEXCAUS', 'KRWUSD=X': 'DEXKOUS', 'MXNUSD=X': 'DEXMXUS',
              'VESUSD=X': 'DEXVZUS', 'INRUSD=X': 'DEXINUS', 'BRLUSD=X': 'DEXBZUS', 'CHFUSD=X': 'DEXSZUS',
              'THBUSD=X': 'DEXTHUS', 'MYRUSD=X': 'DEXMAUS', 'ZARUSD=X': 'DEXSFUS', 'TWDUSD=X': 'DEXTAUS',
              'HKDUSD=X': 'DEXHKUS', 'SGDUSD=X': 'DEXSIUS', 'SEKUSD=X': 'DEXSDUS', 'NOKUSD=X': 'DEXNOUS',
              'DKKUSD=X': 'DEXDNUS'}
    for key, val in fx_map.items():
        if key in df.columns:
            df[key] = df[key].fillna(1 / df[val])
    df['EURUSD=X'] = df['EURUSD=X'].fillna(df['DEXUSEU'])
    df['GBPUSD=X'] = df['GBPUSD=X'].fillna(df['DEXUSUK'])

    # clean up and forward-fill the remaining missing fx rates
    df = df.drop(columns=fred_fx).ffill().rename(columns={'Date': 'date'})
    df.to_csv(os.path.join(OUT_DIR, 'fx_rates.csv'), index=False)


def main():
    # set script start time
    start_time = time.time()

    # run functions
    yahoo_fundamental()
    yahoo_market()
    yahoo_exchange_rates()

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == "__main__":
    main()
