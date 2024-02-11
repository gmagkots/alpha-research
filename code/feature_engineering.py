import os
import sys
import time
import json
import numpy as np
import pandas
import pandas as pd
import signals as sg
import accounting_ratios as ar
import auxiliary_functions as aux
from datetime import timedelta
from scipy.special import ndtri

# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def compile_accounting_ratios():
    """
    Compiles the accounting ratios to be used for signal construction. The observation
    frequency varies across firms, depending on their reporting practices.
    """
    # load the fundamental data and initialize the accounting ratios dataframe
    idcols = ['ticker', 'database', 'date']
    funda = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals.parquet'))
    df = funda[idcols].copy()
    funda = funda.groupby('ticker', group_keys=False)

    # value ratios
    df = pd.concat([df, funda.apply(ar.book_per_share)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(ar.earnings_per_share)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(ar.sales_per_share)], axis=1, copy=False)

    # profitability ratios
    df = pd.concat([df, funda.apply(ar.roe)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(ar.roa)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(ar.roic)], axis=1, copy=False)

    # replace infinite values with NaN and save the fundamental ratios file
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_parquet(os.path.join(OUT_DIR, 'stock_accounting_ratios.parquet'))


def compile_signals():
    """
    Constructs the signals using 21 (252) working days per month (year).
    Avoid the memory-costly pandas merge, use pandas concat instead.
    """
    # load and merge the fundamental ratios and market data
    idcols = ['ticker', 'database', 'date']
    if os.path.isfile(os.path.join(OUT_DIR, 'stock_daily_fundamentals.parquet')):
        mk_cap_thr = 2000
        filename = os.path.join(OUT_DIR, 'stock_signals_{}.parquet'.format(mk_cap_thr))
        dft = pd.read_json(os.path.join(OUT_DIR, 'tickers_mkt_cap_{}.json'.format(mk_cap_thr)))
        funda = pd.read_parquet(os.path.join(OUT_DIR, 'stock_daily_fundamentals.parquet'),
                                filters=[('ticker', 'in', dft['ticker'])])
    else:
        filename = os.path.join(OUT_DIR, 'stock_daily_fundamentals.parquet')
        funda = pd.read_parquet(os.path.join(OUT_DIR, 'stock_accounting_ratios.parquet'))
        dfprc = pd.read_parquet(os.path.join(OUT_DIR, 'stock_daily_prices_usd.parquet'))
        dfret = pd.read_parquet(os.path.join(OUT_DIR, 'stock_daily_returns.parquet'))
        arcols = [col for col in funda.columns if col not in idcols]
        dfmkt = pd.merge(dfprc, dfret, on=idcols, how='left')
        funda = pd.merge(dfmkt, funda, on=idcols, how='left')
        funda[arcols] = funda[arcols].ffill()
        funda.to_parquet(filename)
        sys.exit('Created file {}, stopping execution. Please rerun.'.format(filename))

    # initialize the signals dataframe and group the input data by ticker
    df = funda[idcols + ['price', 'return']].copy()
    funda = funda.groupby('ticker', group_keys=False)

    # construct the signals and concatenate to their container dataframe
    df = pd.concat([df, funda.apply(sg.signal_momentum)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(sg.signal_value)], axis=1, copy=False)
    df = pd.concat([df, funda.apply(sg.signal_profitability)], axis=1, copy=False)

    # save the signals file
    print('Saving the signals file {}'.format(filename))
    df.to_parquet(filename)
    print('Done')


def normalize_signals():
    """
    Normalizes the signals in the cross-section with rank-based
    inverse normal cdf transform (caution with infinite values).

    An alternative transform is to manually transform the raw
    scores to a uniform U[0,1] instead of rank, followed by the
    inverse normal cdf to yield a Z-score. Truncation/winsorization
    should be applied beforehand.

    Subtract the industry median raw value for each signal to mitigate
    bias toward industries with larger scale for signal scores.
    """
    # load the raw signal data, add industry info and group by date
    mk_cap_thr = 2000
    dft = pd.read_json(os.path.join(OUT_DIR, 'tickers_mkt_cap_{}.json'.format(mk_cap_thr)))
    df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_signals_2000.parquet'),
                         filters=[('ticker', 'in', dft['ticker'])])
    df = pd.merge(df, dft, on='ticker', how='left')
    scols = [col for col in df.columns if col.startswith('sg_')]

    # subtract the industry median from each signal per date
    df[scols] = df.groupby('industry')[scols].transform(lambda x: x - x.median())

    # normalize features in the cross-section with rank-based
    # inverse normal cdf transform (caution with infinite values)
    df[scols] = df.groupby('date', sort=False)[scols] \
        .rank(ascending=True, pct=True).apply(ndtri).replace([np.inf, -np.inf], np.nan)

    # # alternative transformation that preserves the raw data sampling distribution shape
    # # (this groupby command changes the sorted order)
    # idcols = ['ticker', 'database', 'price', 'return']
    # df[scols] = df[scols].apply(aux.remove_outliers)
    # df = df.set_index(idcols).groupby('date', sort=False)[scols].apply(aux.uniform_int_transform).reset_index()

    # save the transformed signals
    df.to_parquet(os.path.join(OUT_DIR, 'stock_signals_transformed.parquet'))


def main():
    # set script start time
    start_time = time.time()

    # run functions
    # compile_accounting_ratios()
    # compile_signals()
    normalize_signals()

    # df = pd.read_parquet(os.path.join(DATA_DIR, 'stock_signals_554.parquet'),
    #                      filters=[('ticker', 'in', ["000070.KS", "000270.KS", "001040.KS"])])

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == "__main__":
    main()
