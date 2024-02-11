import os
import sys
import json
import time
import operator
import numpy as np
import pandas as pd
import auxiliary_functions as aux
from itertools import groupby
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import yfinance as yf
import pandas_datareader.data as web

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

DATA_DIR = 'C:\\IBKR TWS API\\my_data\\wrds\\data'
OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def ff_risk_free_rate():
    """
    Downloads and saves the daily risk-free rate from Fama-French library.
    """
    # set the start/end dates and output file name
    start, end = '2000-01-01', (datetime.today() - BDay()).strftime('%Y-%m-%d')
    filename = os.path.join(OUT_DIR, 'external', 'ff_rf.csv')

    # download and save the data
    ffdict = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start, end=end)
    ffdict[0].rename(columns={'RF': 'rf'})['rf'].to_csv(filename, index_label = 'date')


def count_consecutive_values(s, op='==', x=np.nan):
    """
    Function argument sequence imitates filtering condition (s operator x).
    Filter series s to find the max number of consecutive values
    that satisfy a condition (based on operator op and threshhold x).
    Replace all series values that satisfy the condition with -99 and
    then count the frequencies in each group of consecutive -99's.

    Start from the first non-missing value for the series. Special
    case for x = NaN, because itertools groupby doesn't work with NaNs.

    https://stackoverflow.com/questions/68497283/how-to-get-max-count-of-consecutive-1-in-column-pandas
    https://stackoverflow.com/questions/18591778/how-to-pass-an-operator-to-a-python-function

    :param s: input series
    :param op: operator to compare series values with x
    :param x: scalar threshold used to compare with each value in series
    :return: max number of consecutive values that satisfy the desired condition
    """
    # restrict series between first and last non-missing values,
    # use a copy to avoid spillage of -99 into the actual data
    s = s[s.first_valid_index():s.last_valid_index()+1].copy()

    # set operator structure
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq}

    # if x = NaN then replace NaNs in series with -99,
    # else apply the condition to series values
    if np.isnan(x):
        s[s.isna()] = -99
    else:
        s[ops[op](s, x)] = -99

    # list all subgroups that satisfy the condition or get
    # max of consecutive values (length of largest subgroup)
    subgroups = list(sum(g) / -99 for k, g in groupby(s) if k == -99)

    # get max of consecutive values (equal to the length of the largest subgroup),
    # set to zero if list is empty (no consecutive values that satisfy condition)
    res = max(subgroups) if subgroups else 0

    return res


def filter_raw_prices():
    """
    Applies the following filters:
        a) remove stocks with non-positive prices,
        b) remove stocks with insufficient history (3 years, i.e. roughly 756 non-missing values),
        c) remove stocks with missing prices for 10 consecutive days or more,
        d) [HALTED]: remove penny stocks (price < 10 for more than 20% of total history),
        e) remove stocks with stale prices (zero return) for 10 consecutive days or more,
        f) remove outlier returns.
    """
    # loop over databases
    prcs, rets = pd.DataFrame(), pd.DataFrame()
    for base in ['namr', 'global']:
        # load the price data (in USD across all stocks)
        print('Filtering {} data'.format(base))
        df = pd.read_parquet(os.path.join(DATA_DIR, 'compustat_{}_prices_usd.parquet'.format(base)))

        # remove stocks with at least one non-positive price value
        df = df.set_index('Date')
        _, bad_tickers = np.where(df <= 0)
        df = df.drop(columns=df.columns[bad_tickers].unique()).reset_index()

        # remove stocks with insufficient history (less than 756 business days)
        bad_stocks = df.columns[df.apply(lambda x: x.count() < 756)].tolist()
        df = df.drop(columns=bad_stocks)

        # remove stocks with 10 or more consecutive missing prices
        max_consec = df.apply(count_consecutive_values, args=('==', np.nan))
        bad_stocks = max_consec[max_consec >= 10].index.tolist()
        df = df.drop(columns=bad_stocks)

        # remove penny stocks but restore those with most recent price above $50
        # (penny stock criterion might be too strict, it removes established firms
        # in tech industry because of multiple stock splits since 2000)
        dft = df.drop(columns=['Date'])
        penny_freq = dft.apply(lambda prc: (prc < 10).sum() / prc.count())
        valid_stocks = penny_freq[penny_freq <= 0.2].index.tolist()
        last_prc = dft.iloc[-1, :]
        last_prc = last_prc[last_prc >= 50]
        valid_stocks = sorted(list(set(valid_stocks + last_prc.index.tolist())))
        df = df[['Date'] + valid_stocks]

        # construct the daily stock returns and remove their outliers, this may
        # help mitigating signal bias on microcaps that are more volatile
        dfr = df.set_index('Date').pct_change().apply(aux.remove_outliers, action='winsorize').reset_index()

        # remove stocks with stale prices (zero returns) for 10 consecutive days or more
        max_consec = dfr.apply(count_consecutive_values, args=('==', 0))
        bad_stocks = max_consec[max_consec >= 10].index.tolist()
        df = df.drop(columns=bad_stocks)
        dfr = dfr.drop(columns=bad_stocks)

        # return summary statistics
        percentiles = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
        summ = dfr.drop(columns=['Date']).describe(percentiles=percentiles).T.drop(columns=['count'])
        summ.to_csv(os.path.join(OUT_DIR, 'summary stat', '{}_summ_returns_usd.csv'.format(base)),
                    index_label='ticker')
        print(summ[['min', 'max']].describe(percentiles=percentiles))

        # convert the price and return data from wide to long format and concatenate
        dfp = pd.melt(df, id_vars='Date', var_name='ticker', value_name='price')
        dfr = pd.melt(dfr, id_vars='Date', var_name='ticker', value_name='return')
        dfp['database'], dfr['database'] = base, base
        dfp = dfp[['ticker', 'database', 'Date', 'price']].rename(columns={'Date': 'date'})
        dfr = dfr[['ticker', 'database', 'Date', 'return']].rename(columns={'Date': 'date'})
        prcs = pd.concat([prcs, dfp])
        rets = pd.concat([rets, dfr])

        # # save the output prices and returns
        # rcols = ['ticker', 'date', 'return']
        # print('Saving daily prices and returns for database {}'.format(base))
        # df.to_parquet(os.path.join(DATA_DIR, '{}_daily_prices_usd.parquet'.format(base)))
        # dfr[rcols].to_parquet(os.path.join(DATA_DIR, '{}_daily_returns.parquet'.format(base)))

    # save the concatenated returns
    print('Saving the concatenated daily price and return files')
    prcs.to_parquet(os.path.join(OUT_DIR, 'stock_daily_prices_usd.parquet'))
    rets.to_parquet(os.path.join(OUT_DIR, 'stock_daily_returns.parquet'))


def local_to_usd(df, currency='curcd', exclude=None):
    """
    Convert non-USD fundamentals to USD values for Compustat Global.

    :param df: dataframe with fundamental data in local currency
    :param currency: column label for currency info (annual: curcd, quarterly: curcdq)
    :param exclude: columns to exclude from currency conversion
    :return: dataframe with fundamental data in USD currency
    """
    # identify the columns to be processed
    if exclude is None:
        exclude = []
    cols = [col for col in df.select_dtypes(include=['float64']).columns if col not in exclude]

    # read the fx rate series
    fx = pd.read_csv(os.path.join(DATA_DIR, 'fx_rates.csv')).rename(columns={'Date': 'datadate'})
    fx['datadate'] = pd.to_datetime(fx['datadate'])
    valid_currs = [curr[0:3] for curr in fx.columns[1:]]

    # merge the fx rates, restrict to valid currencies,
    # and create the ticker/currency mapping dictionary
    df = pd.merge(df, fx, on='datadate', how='left')
    df = df[df[currency].isin(valid_currs)]
    fx_map = dict(zip(df[currency], df[currency] + 'USD=X'))

    # convert local prices to usd by multiplying with the fxstr rate on selected columns
    for curcd, fxstr in fx_map.items():
        if fxstr == 'USDUSD=X':
            continue
        df.loc[df[currency] == curcd, cols].multiply(df.loc[df[currency] == curcd, fxstr], axis="index")

    # drop the fx rate columns and return
    drop_cols = fx.columns.tolist()[1:]
    return df.drop(columns=drop_cols)


def ewens_intangible_capital(df):
    """
    Processes the intangible capital values in Ewens et al.
    https://github.com/michaelewens/Intangible-capital-stocks

    :param df: input dataframe with fundamentals
    :return: dataframe updated with knowledge and organizational capital
    """
    # auxiliary functions for organization and knowledge capital growth
    def genkcap(g):
        """
        Be sure the first column of g is desired intangible capital,
        second column is parameter for knowledge depreciation rate,
        third column is current period variable (eg, xrd).
        """
        for i in range(1, len(g)):
            g.iloc[i, 0] = g.iloc[i - 1, 0] * (1 - g.iloc[i, 1]) + g.iloc[i, 2]
        return g

    def genocap(g):
        """
        Be sure the first column of g is desired intangible capital,
        second column is parameter for contribution rate,
        third column is current period variable (eg, sg&a).
        """
        for i in range(1, len(g)):
            g.iloc[i, 0] = g.iloc[i - 1, 0] * 0.8 + g.iloc[i, 2] * g.iloc[i, 1]
        return g

    # parameters for delta_{XRD} from Ewens
    sicg1 = [3714, 3716, 3750, 3751, 3792, 4813, 4812, 4841, 4833, 4832] + list(range(100, 1000)) + list(
        range(2000, 2400)) + list(range(2700, 2750)) + list(range(2770, 2800)) + list(range(3100, 3200)) + list(
        range(3940, 3990)) + list(range(2500, 2520)) + list(range(2590, 2600)) + list(range(3630, 3660)) + list(
        range(3710, 3712)) + list(range(3900, 3940)) + list(range(3990, 4000)) + list(range(5000, 6000)) + list(
        range(7200, 7300)) + list(range(7600, 7700)) + list(range(8000, 8100))
    sicg2 = list(range(2520, 2590)) + list(range(2600, 2700)) + list(range(2750, 2770)) + list(
        range(2800, 2830)) + list(
        range(2840, 2900)) + list(range(3000, 3100)) + list(range(3200, 3570)) + list(range(3580, 3622)) + list(
        range(3623, 3630)) + list(range(3700, 3710)) + list(range(3712, 3714)) + list(range(3715, 3716)) + list(
        range(3717, 3750)) + list(range(3752, 3792)) + list(range(3793, 3800)) + list(range(3860, 3900)) + list(
        range(1200, 1400)) + list(range(2900, 3000)) + list(range(4900, 4950))
    sicg3 = [3622, 7391] + list(range(3570, 3580)) + list(range(3660, 3693)) + list(range(3694, 3700)) + list(
        range(3810, 3840)) + list(range(7370, 7380)) + list(range(8730, 8735)) + list(range(4800, 4900))
    sicg4 = list(range(2830, 2840)) + list(range(3693, 3694)) + list(range(3840, 3860))

    # load and merge the files, fill missing orgCapital and knowCapital
    # with zeros to initialize values for firms not in Ewens' file
    ews = pd.read_csv(os.path.join(OUT_DIR, 'external', 'intangibleCapital_122919.csv'))
    df = pd.merge(df, ews, on=['gvkey', 'fyear'], how='left')
    df[['orgCapital', 'knowCapital']] = df[['orgCapital', 'knowCapital']].fillna(0)

    # add growth parameters according to Ewens
    cond = [df['sic'].isin(sicg1), df['sic'].isin(sicg2), df['sic'].isin(sicg3), df['sic'].isin(sicg4)]
    df['delta_r&d'] = np.select(cond, [0.33, 0.42, 0.46, 0.34], default=0.30)
    df['gamma_org'] = np.select(cond, [0.19, 0.22, 0.44, 0.49], default=0.34)

    # evaluate the organization and knowledge capital growths
    df['orgcap'] = df.groupby('gvkey', as_index=False, group_keys=False)[['orgCapital', 'gamma_org', 'sga']].apply(
        genocap)['orgCapital']
    df['knowcap'] = df.groupby('gvkey', as_index=False, group_keys=False)[['knowCapital', 'delta_r&d', 'xrd']].apply(
        genkcap)['knowCapital']
    df[['lag_orgcap', 'lag_knowcap']] = df.groupby('gvkey')[['orgcap', 'knowcap']].shift()

    return df.drop(columns=['orgCapital', 'knowCapital', 'delta_r&d', 'gamma_org'])


def get_fundamentals():
    """
    Filters the annual and Compustat files.
    """
    # restrict to list of IDs where returns are available
    ids, comp = pd.DataFrame(), pd.DataFrame()
    idsfile = os.path.join(OUT_DIR, 'stock_ids.csv')
    for base in ['namr', 'global']:
        tics = pd.read_csv(os.path.join(OUT_DIR, 'summary stat',
                                        '{}_summ_returns_usd.csv'.format(base)), usecols=['ticker'])
        df = pd.read_csv(os.path.join(DATA_DIR, 'compustat_{}_ids.csv'.format(base)), usecols=['gvkey', 'ticker'])
        df = df[df['ticker'].isin(tics['ticker'])]
        df['database'] = base
        ids = pd.concat([ids, df])
        ids.to_csv(idsfile, index=False)
    print('Saved IDs file:', idsfile)

    # import fundamentals from annual and quarterly data
    for freq in ['a', 'q']:
        # reset dataframe across frequencies
        comp = pd.DataFrame()

        # concatenate the data across geographies and restrict to valid stocks only
        for base in ['namr', 'global']:
            # read all fundamental variables and restrict to stocks with return data
            df = pd.read_parquet(
                os.path.join(DATA_DIR, 'filtered_compustat_{}_fund{}_202305.parquet'.format(base, freq)))
            df = df[df['gvkey'].isin(ids['gvkey'].unique())].dropna(axis='columns', how='all')

            if base == 'global':
                # convert float fundamental values from local into USD currency
                # (namr values are in USD, even if the data contain Canadian firms)
                currency = 'curcd' if freq == 'a' else 'curcdq'
                with open(os.path.join(OUT_DIR, 'compustat_int_cols_with_nan.json'), 'r') as f:
                    int_cols_with_nan = json.load(f)
                df = local_to_usd(df, currency=currency, exclude=int_cols_with_nan)

                # global funda shares number labels have an extra "i",
                # rename cshoi -> csho and ajexi -> ajex
                if freq == 'a':
                    df = df.rename(columns={'cshoi': 'csho', 'ajexi': 'ajex'})
            else:
                df = df.drop(columns=['tic'])

            # forward fill the SIC values and update the joint fundamentals frame
            df['database'], df['sic'] = base, df.groupby('gvkey')['sic'].ffill().fillna(0).astype(int)
            comp = pd.concat([comp, df])
        comp = pd.merge(comp, ids, on=['database', 'gvkey'], how='left').dropna(subset=['ticker'])

        # remove columns with only a few scattered zeros
        tot = comp.select_dtypes(include=['number']).sum()
        comp = comp.drop(columns=tot[tot == 0].index.tolist())

        # set certain missing values to zero
        prefix = ('am', 'cogs', 'gdwl', 'intan', 'rdip', 'xrd', 'xsga')
        cols = [col for col in comp.columns if col.startswith(prefix)]
        comp[cols] = comp[cols].fillna(0)

        # correct for Selling General and Administrative (SGA) as in Peters and Taylor (2017)
        if freq == 'a':
            comp['sga'] = comp['xsga'] - comp['xrd'] - comp['rdip']
            cond = [comp['xsga'] == 0, ((comp['xsga'] <= comp['xrd']) & (comp['xrd'] <= comp['cogs']))]
            comp['sga'] = np.select(cond, [0, comp['xsga']], default=comp['sga'])

            # internal knowledge and organizational capital only for the annual files
            comp = ewens_intangible_capital(comp)
        else:
            comp['sgaq'] = comp['xsgaq'] - comp['xrdq'] - comp['rdipq']
            cond = [comp['xsgaq'] == 0, ((comp['xsgaq'] <= comp['xrdq']) & (comp['xrdq'] <= comp['cogsq']))]
            comp['sgaq'] = np.select(cond, [0, comp['xsgaq']], default=comp['sgaq'])

            comp['sgay'] = comp['xsgay'] - comp['xrdy'] - comp['rdipy']
            cond = [comp['xsgay'] == 0, ((comp['xsgay'] <= comp['xrdy']) & (comp['xrdy'] <= comp['cogsy']))]
            comp['sgay'] = np.select(cond, [0, comp['xsgay']], default=comp['sgay'])

        # rearrange rows and save the output
        cols, strfreq = ['gvkey', 'ticker', 'database', 'datadate'], 'annual' if freq == 'a' else 'quarterly'
        filename = os.path.join(OUT_DIR, 'stock_fundamentals_{}.parquet'.format(strfreq))
        comp = comp[cols + [col for col in comp.columns if col not in cols]].rename(columns={'datadate': 'date'})
        print('Saving fundamentals file:', os.path.abspath(filename))
        comp.to_parquet(filename)


def merge_funda_fundq():
    """
    Merges Compustat's annual and quarterly files and saves the common columns between them.
    Caution: can't run on 12GB RAM.
    """
    # read the annual and quarterly files
    funda = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals_annual.parquet'))
    fundq = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals_quarterly.parquet'))

    # select the fundq columns that are also in funda but with slightly different label
    qcols = {col[0:-1] for col in fundq.columns if col.endswith('q')}
    ycols = {col[0:-1] for col in fundq.columns if col.endswith('y')}
    qcols = {col for col in qcols if col in funda.columns}
    ycols = {col for col in ycols if col in funda.columns}
    bcols = qcols.intersection(ycols)

    # saves the common columns between the annual and quarterly files
    out = {"annual and quarterly (2 cols in fundq)": sorted(list(bcols)),
           "annual only (fundq col = funda + 'y')": sorted(list(ycols.difference(bcols))),
           "quarterly only  (fundq col = funda + 'q')": sorted(list(qcols.difference(bcols)))}
    with open(os.path.join(OUT_DIR, 'compustat_funda_fundq_common_cols.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # rename the annual and "quarterly only" fundq columns into their funda equivalents,
    # maintain the quarterly column among the "annual and quarterly" group in fundq
    # (e.g. fundq net income: niy -> combine with ni in funda, niq -> maintain as niq)
    # ycols, qcols = ycols.difference(bcols), qcols.difference(bcols)
    qcols = qcols.difference(bcols)
    ycols, qcols = {(col + 'y'):col for col in ycols}, {(col + 'q'):col for col in qcols}
    fundq = fundq.rename(columns=(ycols | qcols))

    # combine the two files and fill in interim values where possible
    idcols = ['gvkey', 'ticker', 'database', 'date']
    df = pd.concat([funda, fundq]).sort_values(by=idcols)
    df = df.groupby(['gvkey', 'ticker', 'database'], group_keys=True).apply(lambda x: x.ffill())
    df = df.drop_duplicates(subset=idcols, keep='first').reset_index(drop=True)

    # save the output
    filename = os.path.join(OUT_DIR, 'stock_fundamentals.parquet')
    print('Saving fundamentals file:', os.path.abspath(filename))
    df.to_parquet(filename)


def filter_industries():
    """
    Attempts to further restrict the stock universe to a more manageable
    subset for small computing resources.
    """
    # read the filtered fundamentals file and discard some SICs
    df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals_annual.parquet'))
    print('Full number of stocks = {}'.format(len(df['ticker'].unique())))
    df = df[~df['sic'].isin(list(range(4900, 5000)) + list(range(6000, 7000)) + list(range(9000, 10000)))]

    # restrict to one row per company, collect the most frequent SICs and add some manually
    df = df.groupby('gvkey', as_index=False).last()
    # df = df[df['database'] == 'namr']
    sics = df['sic'].value_counts().head(30).index.tolist() + [3711, 3663, 5331, 5961]
    # sics = [sic for sic in sics if sic not in [2836, 5812, 3845, 3559]]
    df = df[df['sic'].isin(sics)]
    # df['sic'].value_counts().to_csv(os.path.join(DATA_DIR, 'temp.csv'))

    # sample randomly up to nmax tickers per SIC to further reduce
    # the sample and restore manually certain key tickers
    nmax, fn = 25, lambda x: x.sample(min(nmax, len(x)))
    key_tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'WMT', 'YELP']
    df = df[['sic', 'ticker']].groupby('sic').apply(fn).reset_index(drop=True)
    tickers = sorted(set(df['ticker'].tolist() + key_tickers))
    print('Total companies =', len(tickers))

    # save the tickers for the selected industries
    with open(os.path.join(OUT_DIR, 'tickers.json'), 'w') as f:
        json.dump(tickers, f, indent=2)


def stock_market_cap():
    """
    Downloads and saves the most recent values of market cap for a list of tickers.
    """
    # # read the fundamentals file to extract the full sample of eligible tickers
    # df = pd.read_parquet(os.path.join(DATA_DIR, 'stock_fundamentals.parquet'))
    # tickers = df['ticker'].unique()
    # print('Total number of stocks:', len(tickers))
    #
    # # download the market cap info from yfinance
    # mktcaps = []
    # for ticker in tickers:
    #     print(ticker)
    #     try:
    #         info = yf.Ticker(ticker).fast_info
    #         marketcap, curr = info['market_cap'], info['currency']
    #         mktcaps.append([ticker, marketcap, curr])
    #     except:
    #         print('No market cap for', ticker)
    #
    # # save the tickers and market cap (in local currency) for post-processing
    # with open(os.path.join(DATA_DIR, 'backup', 'tickers_market_cap_local.json'), 'w') as f:
    #     json.dump(mktcaps, f, indent=2)

    # exclude stocks in regulated utilities (SIC 4900-4999), financial (SIC 6000-6999),
    # and public service industries (SIC 9000+), and those without industry code
    funda = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals.parquet'), columns=['ticker', 'sic'])
    funda = funda.drop_duplicates(keep='last')
    funda = funda[~((funda['sic'].between(4900, 4999)) | (funda['sic'].between(6000, 6999)) |
                    (funda['sic'] >= 9000) | (funda['sic'] == 0))]

    # merge with Fama-French industry classifications
    filename = os.path.join(OUT_DIR, 'external', 'sic_to_ff_industry.csv')
    funda = pd.merge(funda, pd.read_csv(filename), on='sic', how='left')
    funda = funda[~funda['ff_48'].isna()].rename(columns={'ff_48': 'industry'})

    # read the most recent fx rates
    fx = pd.read_csv(os.path.join(DATA_DIR, 'fx_rates.csv'))
    fx = fx.drop(columns=['Date']).iloc[-1].rename('fx')
    fx.index = [x.replace('USD=X', '') for x in fx.index]
    fx['USD'] = 1
    fx.index.names = ['currency']
    fx = fx.reset_index()

    # read the market cap info in local currency and restrict to valid tickers
    with open(os.path.join(OUT_DIR, 'backup', 'tickers_market_cap_local.json'), 'r') as f:
        mktcaps = json.load(f)
    df = pd.DataFrame(mktcaps, columns=['ticker', 'market cap', 'currency'])
    df = pd.merge(df, funda, on='ticker', how='inner')

    # merge the exchange rates with market cap and convert to USD
    # (caution: "GBp" and "ZAc" are 0.01 "GBP" and "ZAR" respectively)
    df['currency'] = df['currency'].str.replace('ZAc', 'ZAR').str.replace('GBp', 'GBP')
    df = pd.merge(df, fx, on='currency', how='left')
    df['market cap usd'] = df['market cap'] * df['fx']
    df.loc[df['currency'].isin(['GBP', 'ZAR']), 'market cap usd'] *= 0.01

    # restrict the estimation universe by sorting on market cap
    # (also option to reduce stocks by choosing evenly among the largest firms)
    mk_cap_thr = 2000
    key_tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'WMT', 'YELP']
    df = df.sort_values('market cap usd', ascending=False)
    df['rank'] = df['market cap usd'].rank(method='max', ascending=False)
    df = df.loc[(df['rank'] <= mk_cap_thr) | (df['ticker'].isin(key_tickers)), ['ticker', 'industry']]  # .iloc[::5, :]

    # drop industries with a single firm in the sample
    df = df.groupby('industry').filter(lambda x: len(x) >= 2)

    # save the tickers
    filename = os.path.join(OUT_DIR, 'tickers_mkt_cap_{}.json'.format(mk_cap_thr))
    print('Saving {} stocks at {}'.format(len(df), os.path.abspath(filename)))
    df.reset_index(drop=True).to_json(filename, orient='records', indent=2)


def main():
    # set script start time
    start_time = time.time()

    # run functions
    # ff_risk_free_rate()
    # filter_raw_prices()
    # get_fundamentals()
    # merge_funda_fundq()
    # filter_industries()
    stock_market_cap()

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == '__main__':
    main()
