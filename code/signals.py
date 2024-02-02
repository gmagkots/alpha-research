import sys
import numpy as np
import pandas as pd
from scipy.special import ndtri


def signal_ensemble(df, signals, label='agg'):
    """
    Aggregates all signals while enforcing a common rank type (i.e. same "units").
    A simple choice is to turn every signal into a Z-score.
    Normalize each signal across all stocks during a date.
    This normalization is suitable to avoid injecting look-ahead bias in the signals.
    Construct the aggregate signal as an equal-weight average across all signals for
    a particular stock and date (i.e. for each row in the dataframe).

    :param df: dataframe with standalone signals
    :param signals: list with the signal (column) names in the dataframe
    :param label: string label for the aggregate signal
    :return: composite signal series
    """
    # standardize the signals but don't replace their original values
    dft = df.set_index('ticker').groupby('date', group_keys=True)[signals]
    dft = dft.rank(ascending=True, pct=True).apply(ndtri).replace([np.inf, -np.inf], np.nan)

    # construct the composite signal as equal-weight average of signal Z-scores
    # and merge the aggregated signal to the original dataframe
    dft[label] = dft[signals].mean(axis=1)
    df = pd.merge(df, dft[['ticker', 'date', label]], on=['ticker', 'date'])

    return df


def signal_annual_growth(df):
    """
    Signal 12-month percent growth (suffix _g12). This implies that the
    signal's growth is the actual predictor, rather than the signal's value.
    Use 252 business days to approximate the annual growth.

    Also add the negative and inverse growth for testing the hypothesis that
    slower, more sustainable growth firms (or maybe focusing on those firms
    with more reliable accounting practices) is the proper signal.

    :param df: dataframe with signals
    :return: dataframe with the signal extensions
    """
    # separate the signal columns and estimate their annual growth,
    # add a suffix to each signal's label
    scols = [col for col in df.columns if col.startswith('sg_')]
    slabels = [sig + '_g12' for sig in scols]
    df[slabels] = df[scols].pct_change(periods=252).replace([np.inf, -np.inf], np.nan)

    # add the negative and inverse growth
    nlabels = [sig + '_g12neg' for sig in scols]
    ilabels = [sig + '_g12inv' for sig in scols]
    df[nlabels] = - df[slabels]
    df[ilabels] = (1 / df[slabels]).replace([np.inf, -np.inf], np.nan)

    return df


def signal_momentum(df):
    """
    x-month momentum signal with daily data (x in [2, 4, 6, 12]):
    lag the return data to exclude most recent month (reversion)
    and form an (x-1)-month rolling window that captures momentum
    from (t-x) to (t-1) months. See also
    https://github.com/mk0417/momentum/blob/master/momentum.py

    Notice that df.shift(periods=12, freq="M") schemas are really
    hard to work with pandas rolling (t-12) to (t-1), because of
    business date and leap year challenges that imply a varying
    window. However, pandas rolling requires a constant window.

    :param df: dataframe with market and fundamental data
    :return: dataframe with signal values
    """
    mdays = [21 * (x - 1) for x in [2, 4, 6, 12]]
    lag_ret = df['return'].shift(21).ffill()
    for days in mdays:
        mlabel = str(int(days / 21 + 1))
        df['sg_mom_m' + mlabel] = (lag_ret.rolling(window=days)
                                   .apply(lambda x: (1 + x).prod()) - 1)
    scols = [col for col in df.columns if col.startswith('sg_mom_m')]
    return df[scols]


def signal_value(df):
    """
    Value signals: book/market, earnings/market, sales/market.
    Divide each accounting value by price to get the proper ratio.

    :param df: dataframe with market and fundamental data
    :return: dataframe with signal values
    """
    # divide by price to create the value signals
    fcols = ['bps', 'bps_orgknow', 'eps', 'eps_orgknow', 'sales_per_share', 'sales_per_share_orgknow']
    scols = ['book/mkt', 'book/mkt_orgknow', 'earn/mkt', 'earn/mkt_orgknow', 'sales/mkt', 'sales/mkt_orgknow']
    scols = ['sg_val_' + sig for sig in scols]
    df[scols] = df[fcols].div(df['price'], axis=0).replace([np.inf, -np.inf], np.nan)

    # # add the signal extensions and update the list of signal labels
    # df = signal_annual_growth(df[['ticker', 'database', 'date'] + scols])
    scols = [col for col in df.columns if col.startswith('sg_val_')]

    return df[scols]


def signal_profitability(df):
    """
    Profitability signals: ROA, ROE, ROIC.

    :param df: dataframe with market and fundamental data
    :return: dataframe with signal values
    """
    # identify and rename the profitability signal labels
    fcols = [col for col in df.columns if col[0:4] in ['roa_', 'roe_', 'roic']]
    df = df.rename(columns={col:'sg_prof_'+col for col in fcols})

    # # add the signal extensions and update the list of signal labels
    # df = signal_annual_growth(df)
    scols = [col for col in df.columns if col.startswith('sg_prof_')]

    return df[scols]

