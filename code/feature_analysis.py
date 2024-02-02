import os
import sys
import time
import json
import numpy as np
import pandas as pd
import auxiliary_functions as aux
from datetime import timedelta

# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def summary_statistics():
    """
    Revisits accounting ratios to summarize and transform if needed.
    """
    # read the accounting ratio or signal data for the stocks in the final sample
    # tickers_json: 'tickers_554' or 'tickers_largest_500'
    # signals_file: 'stock_signals_554' or 'stock_signals_largest_500'
    tickers_json = 'tickers_spread_mkt_cap'
    signals_file = 'stock_signals_spread_mkt_cap'
    print('Estimating summary statistics')
    # with open(os.path.join(OUT_DIR, '{}.json'.format(tickers_json)), 'r') as f:
    #     tickers = json.load(f)
    # df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_accounting_ratios.parquet'),
    #                      filters=[('ticker', 'in', tickers)])
    # scols = [col for col in df.columns if col[0:3] in ['bps', 'eps', 'sal', 'roa', 'roe', 'roi']]
    df = pd.read_parquet(os.path.join(OUT_DIR, '{}.parquet'.format(signals_file)))
    scols = [col for col in df.columns if col.startswith('sg_')]

    # Table 1: full-sample across all firms
    prcs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    dft = df[scols].apply(pd.Series.autocorr).rename('AR(1)')
    summ = pd.concat([df[scols].describe(percentiles=prcs).T, dft], axis=1)
    print(summ)

    # Table 2: summaries across firms and then aggregated
    prcs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    summ = (df.groupby('ticker')[scols]
            .apply(lambda x: x.describe(percentiles=prcs))
            .reset_index().rename(columns={'level_1':'stat'}))
    dft = pd.concat((df[['ticker', col]].groupby('ticker')
                    .apply(lambda x: x[col].autocorr()).rename(col) for col in scols), axis=1).reset_index()
    dft.insert(loc=1, column='stat', value='AR(1)')
    summ = pd.concat([summ, dft])
    summ['stat'] = pd.Categorical(summ['stat'], ['count', 'mean', 'std', 'min', '1%', '5%', '25%',
                                                 '50%', '75%', '95%', '99%', 'max', 'AR(1)'])
    summ = summ.sort_values(['ticker', 'stat'])

    # save the output
    filename = os.path.join(OUT_DIR, 'summary stat', 'signal_summ_stat.csv')
    print('Saving summary statistics at {}'.format(os.path.abspath(filename)))
    summ.to_csv(filename, index=False)
    # summ.to_latex(buf='table.tex', index=False, float_format="%.2f", longtable=False,
    #               column_format='c' * len(summ.columns), label='tab:summ_stat',
    #               caption='Distribution of raw EPS estimates.')


def characteristics_portfolios():
    """
    Forms the signal-based HML portfolios assuming daily rebalancing.
    """
    # restrict to desired variables
    idcols = ['ticker', 'database', 'date', 'return']

    # read the stock returns and signals
    filename = os.path.join(OUT_DIR, 'stock_signals_554_rank_int.parquet')
    # filename = os.path.join(OUT_DIR, 'stock_signals_spread_mkt_cap.parquet')
    # filename = os.path.join(OUT_DIR, 'stock_orthogonal_signals.parquet')
    print('Reading and transforming the stock signal data from \n{}'.format(os.path.abspath(filename)))
    df = pd.read_parquet(filename)
    scols = [col for col in df.columns if col.startswith('sg_')]

    # lag all signals by one period (dataframe already sorted by ticker and date)
    # to avoid injecting look-ahead bias in the regressions
    df[scols] = df[scols].shift().where(df['ticker'] == (df['ticker'].shift()))

    # get the HML portfolio returns for each signal
    print('Initiating signal-based HML portfolio construction.')
    hcols = [col.replace('sg_', 'hml_') for col in scols]
    s = df.groupby('date').apply(aux.daily_hml_returns, scols)
    hml = pd.DataFrame(np.vstack(s), columns=hcols, index=s.index).reset_index()

    # save the HML returns file
    filename = os.path.join(OUT_DIR, 'hml_returns.parquet')
    print('Saving HML portfolios at {}'.format(os.path.abspath(filename)))
    hml.to_parquet(filename)


def signal_performance():
    """
    Evaluates the predictive power of the signal and prints the results. Also
    evaluates whether the signal is priced by another factor (time series regression).
    """
    # print a message
    print('Evaluating the performance of signals')

    # read the HML portfolio returns and drop the portfolios without non-null values
    # hml = pd.read_parquet(os.path.join(OUT_DIR, 'hml_returns.parquet'),
    #                       filters=[('date', '>=', pd.to_datetime('2005-1-1'))])
    hml = pd.read_parquet(os.path.join(OUT_DIR, 'hml_returns.parquet'))
    bad_hmls = hml.columns[hml.isnull().all(axis=0)].tolist()
    if bad_hmls:
        print('The following signals provide no valid HML returns'
              ' throughout the sample period:\n', bad_hmls, '\n')
        hml = hml.drop(columns=bad_hmls)
    scols = hml.columns.tolist()[1:]

    # # keep only selected signals
    # scols = [col for col in scols if '_g12' not in col]

    # # estimate the signal-based stock expected returns
    # xcols = [col.replace('hml', 'sg') for col in scols]
    # cross_sectional_regressions(type='expected returns', scols=xcols)

    # save and print the feature analysis results
    filename = os.path.join(OUT_DIR, 'summary stat', 'signal_performance.csv')
    summ = aux.signal_predictive_power(hml, scols)
    summ.to_csv(filename, index=False)
    print('Predictive power test results at {}'.format(os.path.abspath(filename)))
    print(summ.round(2))

    # test if the momentum factor prices the signals
    momcols = [col for col in scols if 'mom' in col]
    if momcols:
        scols = [col for col in scols if col not in momcols]
        pricing_col = momcols[-1]
        print('Momentum pricing test results for {}'.format(pricing_col))
        for col in scols:
            aux.ols_regression(hml, endog=col, exog=pricing_col, type='pricing')


def cross_sectional_regressions(type='expected returns', scols=None):
    """
    Performs cross-sectional regressions to either estimate expected stock returns
    based on a signal or to orthogonalize a signal relative to another signal/factor.

    The expected returns regressions in the industry are also referred as
    "characteristics-portfolio method". It involves looping over dates and performing
    a cross-sectional regression. The betas and intercepts that can be used to estimate
    parametric (expected) stock returns for a given period are saved after every regression.
    one can estimate E[r_it] = avg(c1, c2, ..., ct) + avg(B1, B2, ..., Bt) * signal_t.
    But what is the use of these expected returns?

    :param type: regression type ('expected returns' or 'orthogonalization')
    :param scols: list of signal labels to focus the analysis on
    :return: None
    """
    # regression type sanity check and signals list
    if type not in ['orthogonalization', 'expected returns']:
        sys.exit('Regression type acceptable keywords: "orthogonalization" or "expected returns"')
    # if scols is None:
    #     scols = ['sg_mom_m12', 'sg_agg_mom',
    #              'sg_val_book/mkt', 'sg_val_book/mkt_orgknow',
    #              'sg_prof_roic_adm_unadj', 'sg_prof_roic_adm_orgknow_q', 'sg_prof_roic_adm_orgknow_y']
    idcols = ['ticker', 'database', 'date', 'return']

    # read the stock return and signal score data
    # df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_signals_spread_mkt_cap.parquet'), columns=idcols + scols)
    df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_signals_554_rank_int.parquet'), columns=idcols + scols)
    # df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_signals_largest_500.parquet'))
    # df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_orthogonal_signals.parquet'), columns=idcols + scols)
    if scols is None:
        scols = [col for col in df.columns if col.startswith('sg_')]

    # orthogonalize selected signals relative to a benchmark signal
    if type == 'orthogonalization':
        momcols = [col for col in scols if 'mom' in col]
        if not momcols:
            sys.exit('No momentum column was found in the data.')
        scols = [col for col in scols if col not in momcols]
        ortho_signal = momcols[-1]
        print('Orthogonalizing signals relative to {}'.format(ortho_signal))
        for col in scols:
            df[col] = df.groupby('date').apply(aux.ols_regression, endog=col, exog=ortho_signal,
                                               type='orthogonalization').reset_index(level=0, drop=True)
        filename = os.path.join(OUT_DIR, 'stock_orthogonal_signals.parquet')
        print('Saving file {}'.format(os.path.abspath(filename)))
        df.drop(columns=scols+momcols).to_parquet(filename)

    # perform cross-sectional regressions for each signal to estimate
    # and save their expected (parametric) returns. Lag all signals by
    # one period so that each row contains the expected return from each
    # signal along with the realized return
    if type == 'expected returns':
        print('Estimating the signal-based stock expected returns')
        df[scols] = df[scols].shift().where(df['ticker'] == (df['ticker'].shift()))
        for col in scols:
            df[col.replace('sg_', 'Eret_')] = \
                df.groupby('date').apply(aux.ols_regression, endog='return', exog=col,
                                         type='expected returns').reset_index(level=0, drop=True)
        filename = os.path.join(OUT_DIR, 'stock_expected_returns.parquet')
        print('Saving file {}'.format(os.path.abspath(filename)))
        df.drop(columns=scols).to_parquet(filename)


def main():
    # set script start time
    start_time = time.time()

    # run functions
    # summary_statistics()
    # characteristics_portfolios()
    # signal_performance()
    # # cross_sectional_regressions(type='expected returns')
    # cross_sectional_regressions(type='orthogonalization')

    # with open(os.path.join(OUT_DIR, 'tickers_554.json'), 'r') as f:
    #     tickers = json.load(f)
    # df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_fundamentals.parquet'),
    #                      filters=[('ticker', 'in', tickers)])

    idcols = ['ticker', 'database', 'date', 'return']
    scols = ['bps', 'eps', 'roe_hou_orgknow_ebit']
    cols = idcols + scols
    with open(os.path.join(OUT_DIR, 'tickers_554.json'), 'r') as f:
        tickers = json.load(f)
    # tickers = tickers[0:50]
    df = pd.read_parquet(os.path.join(OUT_DIR, 'stock_daily_fundamentals.parquet'), filters=[('ticker', 'in', tickers)])
    # df = df.loc[df['date'].between(pd.to_datetime('2023-05-26'), pd.to_datetime('2023-05-31')), cols]
    df = df.loc[df['date'] == pd.to_datetime('2023-05-31'), cols]



    print(df.tail())
    print(df[scols].describe())

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == "__main__":
    main()
