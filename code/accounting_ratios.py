import numpy as np


'''
###########################################################
### Value
###########################################################
'''
def book_per_share(df):
    """
    Book per share, to be used with price from market data and form B/M ratio.
    See Hou, Xue, and Zhang 2019 anomaly A.2.3 for construction.

    Intangibles extension: consider ALL organizational and knowledge capital as
    part of shareholder's (book) equity. This is the same as the replacement
    cost (book value) of intangible capital in Peters and Taylor 2017 (see eq 9).

    Set book value to NaN for firm-dates that is negative to exclude these stocks
    from portfolio construction at the particular point in time. Negative book
    values imply that the firm may be insolvent. As soon as the book value jumps
    back to positive, the firm will be considered again for portfolio construction.
    """
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    csho = df['csho'] * df['ajex']
    r_orgn = (- 0.2 + 0.3 * df['sga'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrd'] / df['lag_knowcap']).fillna(0)

    # Hou, Xue, and Zhang 2019 anomaly A.2.3 description (and intangibles extension)
    se = np.select([df['seq'].notna(), (df['ceq'] + df['pstk']).notna(), (df['at'] - df['lt']).notna()],
                   [df['seq'], df['ceq'] + df['pstk'], df['at'] - df['lt']], default=np.nan)
    df['bps'] = (se + df['txditc'].fillna(0) - df['pstk']) / csho
    # df['bps_orgknow'] = df['bps'] + (df['knowcap'] + df['orgcap']) / csho
    df['bps_orgknow'] = df['bps'] * (1 + r_orgn + r_know)

    # set ratio to NaN when book value is negative
    df.loc[df['bps'] < 0, ['bps', 'bps_orgknow']] = np.nan

    return df[['bps', 'bps_orgknow']]


def earnings_per_share(df):
    """
    Earnings per share, to be used with price from market data and form E/M ratio.
    See Hou, Xue, and Zhang 2019 anomaly A.2.10 for construction.

    Intangibles extension: since earnings is a form of profit, add the growth
    in organizational and knowledge capital as part of earnings.
    """
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    csho = df['csho'] * df['ajex']
    r_orgn = (- 0.2 + 0.3 * df['sga'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrd'] / df['lag_knowcap']).fillna(0)

    # Hou, Xue, and Zhang 2019 anomaly A.2.10 description (and intangibles extension)
    df['eps'] = df['ibq'] / csho
    # df['eps_orgknow'] = (df['ib'] + df['xrd'] + 0.3 * df['sga']
    #                      - 0.15 * df['knowcap'] - 0.2 * df['orgcap']) / csho
    df['eps_orgknow'] = df['eps'] * (1 + r_orgn + r_know)

    return df[['eps', 'eps_orgknow']]


def sales_per_share(df):
    """
    Earnings per share, to be used with price from market data and form sales/market ratio.
    See Hou, Xue, and Zhang 2019 anomaly A.2.23 for construction.

    Intangibles extension: consider the organizational and knowledge capital as
    a multiplying (augmenting) return to sales, possibly to capture the extra sales
    that the company would have had in the short run if they didn't invest in intangibles
    and to get a more precise forecast of what sales will be once the benefit of
    intangibles materializes into extra sales.
    """
    # Hou, Xue, and Zhang 2019 anomaly A.2.23 description,
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    csho = df['csho'] * df['ajex']
    r_orgn = (- 0.2 + 0.3 * df['sgaq'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrdq'] / df['lag_knowcap']).fillna(0)
    df['sales_per_share'] = df['saleq'] / csho
    df['sales_per_share_orgknow'] = df['sales_per_share'] * (1 + r_orgn + r_know)

    return df[['sales_per_share', 'sales_per_share_orgknow']]


'''
###########################################################
### Profitability
###########################################################
'''
def roa(df):
    """
    Return on assets (ROA):
    Balakrishnan, Bartov and Faurel 2010
    ROA = IB / AT and IBQ / AT
    Variation: add IB during quarter (IBQ) and IB during calendar year (IBY).
    """
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    r_orgn = (- 0.2 + 0.3 * df['sgaq'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrdq'] / df['lag_knowcap']).fillna(0)

    # Balakrishnan, Bartov and Faurel 2010
    lag_at = df['at'].shift()
    df['roa_bbf_q'] = df['ibq'] / lag_at
    df['roa_bbf_y'] = df['ib'] / lag_at

    # intangibles extensions
    lag_assets = (df['at'] * (1 + r_orgn + r_know)).shift()
    adjpr_qi = df['ibq'] + df['xrdq'] + 0.3 * df['sgaq'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    adjpr_yi = df['ib'] + df['xrd'] + 0.3 * df['sga'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    adjpr_qe = (df['ebit'] + df['amq'] + df['xrdq'] + 0.3 * df['sgaq']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    adjpr_ye = (df['ebit'] + df['am'] + df['xrd'] + 0.3 * df['sga']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    df['roa_orgknow_q_ib'], df['roa_orgknow_y_ib'] = adjpr_qi / lag_assets, adjpr_yi / lag_assets
    df['roa_orgknow_q_ebit'], df['roa_orgknow_y_ebit'] = adjpr_qe / lag_assets, adjpr_ye / lag_assets

    cols = ['roa_bbf_q', 'roa_bbf_y',
            'roa_orgknow_q_ib', 'roa_orgknow_y_ib',
            'roa_orgknow_q_ebit', 'roa_orgknow_y_ebit']
    return df[cols]


def roe(df):
    """
    Return on equity (ROE)
    Haugen and Baker 1996
    ROE = NI / CEQ and NIQ / CEQ

    Hou, Xue, and Zhang 2015 (also in HXZ 2019 as anomaly A.4.1):
    ROE = IBQ / (se + TXDITC - PSTK)_{t-1}
    se = SEQ or (CEQ + PSTK) or (AT - LT) in that order
    Variation: add NI during quarter (NIQ) and NI during calendar year (NIY).

    Intangibles extension: consider ALL organizational and knowledge capital as
    part of shareholder's (book) equity.
    """
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    r_orgn = (- 0.2 + 0.3 * df['sgaq'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrdq'] / df['lag_knowcap']).fillna(0)

    # # Haugen and Baker 1996
    # ceq = df['ceq'].shift()
    # df['roe_hb_q'], df['roe_hb_y'] = df['niq'] / ceq, df['ni'] / ceq

    # Hou, Xue, and Zhang 2015 (and intangibles extension)
    se = np.select([df['seq'].notna(), (df['ceq'] + df['pstk']).notna(), (df['at'] - df['lt']).notna()],
                   [df['seq'], df['ceq'] + df['pstk'], df['at'] - df['lt']], default=np.nan)
    lag_equity = (se + df['txditc'].fillna(0) - df['pstk']).shift()
    lag_equity_adj = ((se + df['txditc'].fillna(0) - df['pstk']) * (1 + r_orgn + r_know)).shift()
    adjpri = df['ibq'] + df['xrd'] + 0.3 * df['sga'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    adjpre = df['ebit'] + df['am'] + df['xrd'] + 0.3 * df['sga'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    df['roe_hou'] = df['ibq'] / lag_equity
    df['roe_hou_orgknow_ib'], df['roe_hou_orgknow_ebit'] = adjpri / lag_equity_adj, adjpre / lag_equity_adj

    # return df[['roe_hb_q', 'roe_hb_y', 'roe_hou', 'roe_hou_orgknow']]
    return df[['roe_hou', 'roe_hou_orgknow_ib', 'roe_hou_orgknow_ebit']]


def roic(df):
    """
    Return on invested capital (ROIC):
    Brown and Rowe 2007
    ROIC = (EBIT - NOPI(Q)) / (CEQ + LT + CHE)

    Ayyagari, Demirgüç-Kunt, Maksimovic (RFS, 2023)
    Unadjusted ROIC: equations (1) and (2)
    ROIC adjusted for organizational and intangible capital: eq (3)-(5)
    """
    # intangibles returns stem from dividing equation (11) in
    # Peters and Taylor 2017 by G_{i,t-1}
    r_orgn = (- 0.2 + 0.3 * df['sgaq'] / df['lag_orgcap']).fillna(0)
    r_know = (- 0.15 + df['xrdq'] / df['lag_knowcap']).fillna(0)

    # simple ratios
    lag_cap = df['icapt'].shift()
    lag_cap_adj = (df['icapt'] - df['intan'] + df['knowcap'] + df['orgcap']).shift()
    adjpriq = df['ibq'] + df['xrdq'] + 0.3 * df['sgaq'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    adjpriy = df['ib'] + df['xrd'] + 0.3 * df['sga'] - 0.15 * df['knowcap'] - 0.2 * df['orgcap']
    # df['roic_niq_icapt'], df['roic_ni_icapt'] = df['niq'] / lag_cap, df['ni'] / lag_cap
    df['roic_ibq_icapt'], df['roic_ib_icapt'] = df['ibq'] / lag_cap, df['ib'] / lag_cap
    df['roic_ibq_icapt_orgknow'], df['roic_ib_icapt_orgknow'] = adjpriq / lag_cap_adj, adjpriy / lag_cap_adj

    # Brown and Rowe 2007
    lag_ic = (df['ceq'] + df['lt'] + df['che']).shift()
    lag_ic_adj = ((df['ceq'] + df['lt'] + df['che']) * (1 + r_orgn + r_know)).shift()
    adjpr_q = (df['ebit'] - df['nopiq'] + df['xrdq'] + 0.3 * df['sgaq']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    adjpr_y = (df['ebit'] - df['nopi'] + df['xrd'] + 0.3 * df['sga']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    df['roic_br_q'] = (df['ebit'] - df['nopiq']) / lag_ic
    df['roic_br_y'] = (df['ebit'] - df['nopi']) / lag_ic
    df['roic_br_q_orgknow'], df['roic_br_y_orgknow'] = adjpr_q / lag_ic_adj, adjpr_y / lag_ic_adj

    # Ayyagari, Demirgüç-Kunt, Maksimovic 2023
    inv_cap_unadj = (df['ppent'] + df['act'] + df['intan'] - df['lct'] - df['gdwl']
                     - np.fmax(df['che'] - 0.02 * df['sale'], 0)).shift()
    df['roic_adm_unadj'] = (df['ebit'] + df['am']) / inv_cap_unadj

    adjpr_q = (df['ebit'] + df['amq'] + df['xrdq'] + 0.3 * df['sgaq']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    adjpr_y = (df['ebit'] + df['am'] + df['xrd'] + 0.3 * df['sga']
               - 0.15 * df['knowcap'] - 0.2 * df['orgcap'])
    inv_cap_adj = (df['ppent'] + df['act'] + df['knowcap'] + df['orgcap']
                   - df['lct'] - df['gdwl'] - np.fmax(df['che'] - 0.02 * df['sale'], 0)).shift()
    df['roic_adm_orgknow_q'], df['roic_adm_orgknow_y'] = adjpr_q / inv_cap_adj, adjpr_y / inv_cap_adj

    cols = ['roic_ib_icapt', 'roic_ibq_icapt', 'roic_ib_icapt_orgknow', 'roic_ibq_icapt_orgknow',
            'roic_br_q', 'roic_br_y', 'roic_br_q_orgknow', 'roic_br_y_orgknow',
            'roic_adm_unadj', 'roic_adm_orgknow_q', 'roic_adm_orgknow_y']
    return df[cols]

