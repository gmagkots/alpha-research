import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtri
from statsmodels.iolib.summary2 import summary_params

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def remove_outliers(s, action='truncate'):
    """
    Removes outliers based on the median absolute deviation or
    the mean absolute deviation when the former is zero.

    :param s: input series
    :param action: 'truncate' or 'winsorize'
    :return: series with outliers truncated or winsorized
    """
    # Median Absolute Deviation (if negative, use Mean Absolute Deviation)
    term = s - s.median()
    mad = 1.4826 * term.abs().median()
    if mad < 0:
        mad = 0.7979 * term.abs().mean()
    if action == 'truncate':
        s[term.abs() / mad > 5] = np.nan
    elif action == 'winsorize':
        s[term > 5 * mad] = 5 * mad
        s[term < - 5 * mad] = - 5 * mad
    else:
        sys.exit('Keyword "action" value should be "truncate" or "winsorize".')
    return s


def daily_hml_returns(df, signals):
    """
    Ranks stocks in the cross-section into High and Low groups for each
    signal and outputs the difference in their equal-weighted returns.

    :param df: stock signal and return data reflecting a particular date
    :param signals: iterable with signal labels
    :return: pandas series of lists, each list containing the
             returns of signal-based HML portfolios
    """
    # # transform potential string input for signals into list
    # if isinstance(signals, str):
    #     signals = [signals]

    # rank stocks into signal deciles in the cross section and
    # estimate the HML portfolio returns for each signal
    hmls = []
    for sig in signals:
        try:
            deciles = pd.qcut(df[sig], q=10, labels=False, retbins=False, duplicates='drop') + 1
            ret_lo = df.loc[deciles == 1, 'return'].mean()
            ret_hi = df.loc[deciles == 10, 'return'].mean()
            hmls.append(ret_hi - ret_lo)
        except IndexError:
            hmls.append(np.nan)

    return hmls


def signal_predictive_power(df, signals):
    """
    Evaluates the predictive power of the signal and prints the results.
    Use 756 periods as the 3-year window for rolling SR estimation (3*252 = 765),
    because Pandas rolling function doesn't accept non-fixed window.

    :param df: dataframe with signal and market return series
    :param signals: signal name (must be a column in the input dataframe)
    :return: dataframe with feature analysis statistics
    """
    # merge the market return (S&P 500) and risk-free rate with the signals
    mkt = pd.read_csv(os.path.join(OUT_DIR, 'external', 'snp500_rf.csv'))
    mkt['date'] = pd.to_datetime(mkt['date'])
    df = pd.merge(df, mkt, on='date')

    # full-sample Sharpe and Information Ratios (relative to market factor)
    active_rets = df[signals].subtract(df['snp500'], axis=0)
    sr = df[signals + ['snp500']].mean() / df[signals + ['snp500']].std() * np.sqrt(252)
    ir = active_rets.mean() / active_rets.std() * np.sqrt(252)

    # information ratio (IC) as the TS average of correlations between pairs of
    # expected (parametric) return forecasts from CS regressions and their
    # corresponding stock realized returns in the cross section
    er = pd.read_parquet(os.path.join(OUT_DIR, 'stock_expected_returns.parquet'))
    ercols = [col for col in er.columns if col.startswith('Eret')]
    ic = er.groupby('date').apply(lambda d: d[ercols].corrwith(d['return'])).mean()
    ic.index = [idx.replace('Eret', 'hml') for idx in ic.index]

    # 3-year rolling Sharpe ratios min and max
    dft = df[signals + ['snp500']].rolling(765)
    srs = dft.apply(lambda x: x.mean() / x.std() * np.sqrt(252))
    sr_min_max = srs.agg(['min', 'max'])

    # estimate the drawdowns and the mdd for every time series
    dds = (1 + df[signals + ['snp500']]).cumprod() / (1 + df[signals + ['snp500']]).cumprod().cummax() - 1
    mdd = dds.min()  # single largest drawdown

    # # cumulative return and cumulative sum of returns (no reinvestment)
    # cumrets = (1 + df[signals + ['snp500']]).cumprod() - 1
    # cumsums = df[signals + ['snp500']].cumsum()

    # combine all metrics in a dataframe
    out = pd.concat([sr.rename('Sharpe Ratio (SR)'), ir.rename('Information Ratio (IR)'),
                     ic.rename('Information Coefficient (IC)'),
                     sr_min_max.T.rename(columns={'min': '3-year Rolling SR Min', 'max': '3-year Rolling SR Max'}),
                     mdd.rename('Maximum Drawdown (MDD)')], axis=1)
    out.index.name = 'HML Portfolio'

    return out.reset_index()


def ols_regression(df, endog, exog, type='expected returns'):
    """
    Performs a regression in the cross section of stocks for a specific date.
    The regression can be either for estimating signal-based expected returns
    (y = realized stock returns, x = lagged signal score) or to orthogonalize
    a signal relative to another factor/signal (y = signal score, x = concurrent
    orthogonalizing signal/factor score).

    :param df: input dataframe containing stock-level signal and return data
    :param endog: dependent variable (stock return or signal score)
    :param exog: independent variable (lagged signal or orthogonalizing signal score)
    :param type: regression type ('expected returns', 'orthogonalization', 'pricing')
    :return: series of expected stock returns or signal residuals (signal with its projection
             on the orthogonalizing factor subtracted)
    """
    # # normalize signals into Z-scores before the orthogonalization
    # # (comment out if signals are already in Z-score format from
    # # rank-based inverse normal CDF transformation)
    # if type == 'orthogonalization':
    #     df[endog] = zscore(df[endog], nan_policy='omit', ddof=1)
    #     df[exog] = zscore(df[exog], nan_policy='omit', ddof=1)

    # perform the XS or TS regression
    try:
        res = sm.OLS(df[endog], sm.add_constant(df[exog]), missing='drop').fit()
    except ValueError:
        df[exog] = np.nan
        return df[exog]

    # return what is required depending on type
    if type == 'orthogonalization':
        return res.resid
    elif type == 'expected returns':
        return res.fittedvalues
    elif type == 'pricing':
        print('Signal', endog)
        print(summary_params(res, float_format='%0.3f', use_t=True), '\n')
    else:
        sys.exit('Regression type acceptable values are "expected returns", "orthogonalization", "pricing".')


def uniform_int_transform(s, tmin=0.1, tmax=0.9):
    """
    Transforms raw feature values into uniform U[tmin, tmax]
    and then to a Z-score with the inverse normal cdf. Tmin
    and tmax must be between [0,1].

    This transform maintains the shape of the original sample
    distribution, unlike the rank-based transform that yields
    perfectly bell-shaped Z-scores. This is why it is
    recommended to treat outliers before transforming.

    Example use:
    df[scols] = df[scols].apply(remove_outliers).apply(uniform_int_transform)

    :param s: input series with the raw scores
    :param tmin: lower limit in the uniform distribution
    :param tmax: upper limit in the uniform distribution
    :return: series transformed to Z-scores
    """
    s = (s - s.min()) / (s.max() - s.min()) * (tmax - tmin) + tmin
    return ndtri(s)
