import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import time
from _collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

base_dict = {'symbol': '股票代码', 'name': '股票名称', 'area': '所在地域', 'industry': '所属行业', 'market': '市场类型',
             'list_date': '上市日期', 'setup_date': '注册日期', 'employees': '员工人数'}
indicator_dict = {'ts_code':'ts_code','ann_date':'ann_date','end_date':'end_date','q_eps':'每股收益(单季度)','eps':'每股收益','revenue_ps':'每股营业收入','capital_rese_ps':'每股资本公积','surplus_rese_ps':'每股盈余公积',
                      'current_ratio':'流动比率','quick_ratio':'速动比率','profit_to_gr':'净利润/营业总收入','q_profit_to_gr':'净利润/营业总收入(单季度)','op_of_gr':'营业利润/营业总收入','q_op_to_gr':'营业利润/营业总收入(单季度)','roe':'净资产收益率','q_roe':'净资产收益率(单季度)',
                      'roe_waa':'加权平均净资产收益率','roa_dp':'总资产净利率(杜邦分析)','debt_to_assets':'资产负债率','assets_turn':'总资产周转率','bps':'每股净资产','cfps':'每股现金流',
                      'ocf_to_opincome':'净现比','basic_eps_yoy':'每股收益同比增长率','cfps_yoy':'每股现金流同比增长率','or_yoy':'营业收入同比增长率','q_sales_yoy':'营业收入同比增长率(单季度)',
                      'q_profit_yoy':'净利润同比增长率(单季度)','equity_yoy':'净资产同比增长率','ar_turn':'应收账款周转率','ca_turn':'流动资产周转率','inv_turn':'存货周转率','undist_profit_ps':'每股未分配利润',
                      'tax_to_ebt':'所得税/利润总额'}
income_dict = {'ts_code':'ts_code','end_date':'end_date','revenue':'营业收入','total_profit':'利润总额','income_tax':'所得税'}
balance_dict = {'ts_code':'ts_code','end_date':'end_date','accounts_receiv':'应收账款','fix_assets':'固定资产','total_cur_liab':'流动负债'}
month2season_dict = {'季最高价': 'high', '季最低价': 'low', '季成交量': 'vol','月涨跌幅(复权)_1':'pct_chg_1', '月涨跌幅(复权)_2':'pct_chg_2','月涨跌幅(复权)_0':'pct_chg_0'}

daily2season_dict = {'换手率_mean': 'turnover_rate_mean', '市盈率TTM_amin': 'pe_ttm_amin', '市盈率TTM_mean': 'pe_ttm_mean', '市盈率TTM_amax':'pe_ttm_amax',
                     '市净率_amin':'pb_amin','市净率_mean':'pb_mean', '市净率_amax':'pb_amax',
                    '市销率（TTM）_amin': 'ps_ttm_amin', '市销率（TTM）_mean': 'ps_ttm_mean', '市销率（TTM）_amax': 'ps_ttm_amax',
                    '股息率TTM_mean':'dv_ttm_mean', '总股本(万)_mean':'total_share_mean', '总市值(万)_mean': 'total_mv_mean',
                    '流通股本(万)_mean': 'float_share_mean', '流通市值(万)_mean': 'circ_mv_mean', '波动率':'close_std'}

computed_dict = {'label':'label','label_new':'label_new','float_share_to_total_share':'流通A比例', 'list_time': '上市时间', 'setup_date' : '公司成立时间', 'chg_max_pos':'季最大涨幅','chg_max_neg':'季最大跌幅'}

final_dict = {**base_dict,**indicator_dict,**income_dict, **balance_dict, **computed_dict, **{v:k for (k,v) in month2season_dict.items()}, **{v:k for (k,v) in daily2season_dict.items()}}
json.dump(final_dict, open('convert_dict.json','w',encoding='utf-8'), ensure_ascii=False)

def revise_name():
    #revise_base
    # df = pd.read_sql('select * from stock_basic', engine_ts)
    # base_dict = {'symbol': '股票代码', 'name': '股票名称', 'area': '所在地域', 'industry': '所属行业', 'market': '市场类型',
    #              'list_date': '上市日期', 'setup_date': '注册日期', 'employees': '员工人数'}
    # base_dict_reversed = {v:k for (k,v) in base_dict.items()}
    # df.rename(columns=base_dict_reversed, inplace=True)
    # df.to_sql('stock_basic', engine_ts, index=False, if_exists='replace', chunksize=5000)
    # print('stock basic')
    #
    # #revise season basic
    # df = pd.read_sql('select * from stock_season_basic', engine_ts)
    # indicator_dict = {'ts_code':'ts_code','ann_date':'ann_date','end_date':'end_date','q_eps':'每股收益(单季度)','eps':'每股收益','revenue_ps':'每股营业收入','capital_rese_ps':'每股资本公积','surplus_rese_ps':'每股盈余公积',
    #                   'current_ratio':'流动比率','quick_ratio':'速动比率','profit_to_gr':'净利润/营业总收入','q_profit_to_gr':'净利润/营业总收入(单季度)','op_of_gr':'营业利润/营业总收入','q_op_to_gr':'营业利润/营业总收入(单季度)','roe':'净资产收益率','q_roe':'净资产收益率(单季度)',
    #                   'roe_waa':'加权平均净资产收益率','roa_dp':'总资产净利率(杜邦分析)','debt_to_assets':'资产负债率','assets_turn':'总资产周转率','bps':'每股净资产','cfps':'每股现金流',
    #                   'ocf_to_opincome':'净现比','basic_eps_yoy':'每股收益同比增长率','cfps_yoy':'每股现金流同比增长率','or_yoy':'营业收入同比增长率','q_sales_yoy':'营业收入同比增长率(单季度)',
    #                   'q_profit_yoy':'净利润同比增长率(单季度)','equity_yoy':'净资产同比增长率','ar_turn':'应收账款周转率','ca_turn':'流动资产周转率','inv_turn':'存货周转率','undist_profit_ps':'每股未分配利润',
    #                   'tax_to_ebt':'所得税/利润总额'}
    # income_dict = {'ts_code':'ts_code','end_date':'end_date','revenue':'营业收入','total_profit':'利润总额','income_tax':'所得税'}
    # balance_dict = {'ts_code':'ts_code','end_date':'end_date','accounts_receiv':'应收账款','fix_assets':'固定资产','total_cur_liab':'流动负债'}
    # season_base_dict = {**indicator_dict,**income_dict,**balance_dict}
    # season_base_dict_reversed = {v: k for (k, v) in season_base_dict.items()}
    # df.rename(columns=season_base_dict_reversed, inplace=True)
    # df.to_sql('stock_season_basic', engine_ts, index=False, if_exists='replace', chunksize=5000)
    # print('season basic')
    #
    #
    # # revise stock_monthly
    # df = pd.read_sql('select * from stock_monthly', engine_ts)
    # stock_monthly_dict = {'ts_code': 'ts_code', 'trade_date': '交易日期', 'close': '月收盘价', 'open': '月开盘价', 'high': '月最高价',
    #                 'low': '月最低价', 'change': '月涨跌额', 'pct_chg': '月涨跌幅(复权)', 'vol': '月成交量', 'amount': '月成交额'}
    # stock_monthly_dict_reversed = {v: k for (k, v) in stock_monthly_dict.items()}
    # df.rename(columns=stock_monthly_dict_reversed, inplace=True)
    # df.to_sql('stock_monthly', engine_ts, index=False, if_exists='replace', chunksize=5000)
    # print('stock_monthly')
    #
    #
    # # revise stock daily
    # df = pd.read_sql('select * from stock_daily', engine_ts)
    # daily_dict = {'ts_code': 'ts_code', 'trade_date': '交易日期', 'close': '当日收盘价', 'turnover_rate': '换手率',
    #              'pe_ttm': '市盈率TTM', 'pb': '市净率', 'ps_ttm': '市销率（TTM）', 'dv_ttm': '股息率TTM',
    #              'total_share': '总股本(万)', 'float_share': '流通股本(万)', 'total_mv': '总市值(万)', 'circ_mv': '流通市值(万)'}
    # daily_dict_reversed = {v: k for (k, v) in daily_dict.items()}
    # df.rename(columns=daily_dict_reversed, inplace=True)
    # df.to_sql('stock_daily', engine_ts, index=False, if_exists='replace', chunksize=5000)
    # print('stock_daily')
    #
    #
    #
    # # revise month2season
    # df = pd.read_sql('select * from stock_month2season', engine_ts)
    # month2season_dict = {'季最高价': 'high', '季最低价': 'low', '季成交量': 'vol','月涨跌幅(复权)_1':'pct_chg_1',
    #                      '月涨跌幅(复权)_2':'pct_chg_2','月涨跌幅(复权)_0':'pct_chg_0'}
    # df.rename(columns=month2season_dict, inplace=True)
    # df.to_sql('stock_month_to_season', engine_ts, index=False, if_exists='replace', chunksize=5000)
    # print('stock_month_to_season')


    #revise daily2season
    df = pd.read_sql('select * from stock_day2season', engine_ts)
    daily2season_dict = {'换手率_mean': 'turnover_rate_mean', '市盈率TTM_amin': 'pe_ttm_amin', '市盈率TTM_mean': 'pe_ttm_mean',
                         '市盈率TTM_amax': 'pe_ttm_amax',
                         '市净率_amin': 'pb_amin', '市净率_mean': 'pb_mean', '市净率_amax': 'pb_amax',
                         '市销率（TTM）_amin': 'ps_ttm_amin', '市销率（TTM）_mean': 'ps_ttm_mean', '市销率（TTM）_amax': 'ps_ttm_amax',
                         '股息率TTM_mean': 'dv_ttm_mean', '总股本(万)_mean': 'total_share_mean',
                         '总市值(万)_mean': 'total_mv_mean',
                         '流通股本(万)_mean': 'float_share_mean', '流通市值(万)_mean': 'circ_mv_mean'}
    df.rename(columns=daily2season_dict, inplace=True)
    df.to_sql('stock_daily_to_season', engine_ts, index=False, if_exists='replace', chunksize=5000)
    print('stock_daily_to_season')


def get_basic_info():
    base_dict = {'symbol': '股票代码', 'name': '股票名称', 'area': '所在地域', 'industry': '所属行业', 'market': '市场类型',
                 'list_date': '上市日期', 'setup_date': '注册日期', 'employees': '员工人数'}
    df_base = pro.stock_basic()
    ts_code_lst = df_base['ts_code'].tolist()
    company_lst = []
    for i in range(len(df_base)//100+1):
        if i != len(df_base)//100:
            tmp_company = pro.stock_company(ts_code=','.join(ts_code_lst[i*100:(i+1)*100]), fields='ts_code,setup_date,employees')
        else:
            tmp_company = pro.stock_company(ts_code=','.join(ts_code_lst[i*100:]), fields='ts_code,setup_date,employees')
        company_lst.append(tmp_company)
        time.sleep(12)

    df_company = pd.concat(company_lst)
    df = pd.merge(df_base,df_company,how='left',on='ts_code')
    # df.rename(columns=base_dict, inplace=True)
    # res = df.to_sql('stock_basic', engine_ts, index=False, if_exists='append', chunksize=5000)
    return df

def get_season_basic(df_base, start_date='20160101', end_date='20201231', replace=False):
    # start_date, end_date = '20160101', '20201231'
    indicator_dict = {'ts_code':'ts_code','ann_date':'ann_date','end_date':'end_date','q_eps':'每股收益(单季度)','eps':'每股收益','revenue_ps':'每股营业收入','capital_rese_ps':'每股资本公积','surplus_rese_ps':'每股盈余公积',
                      'current_ratio':'流动比率','quick_ratio':'速动比率','profit_to_gr':'净利润/营业总收入','q_profit_to_gr':'净利润/营业总收入(单季度)','op_of_gr':'营业利润/营业总收入','q_op_to_gr':'营业利润/营业总收入(单季度)','roe':'净资产收益率','q_roe':'净资产收益率(单季度)',
                      'roe_waa':'加权平均净资产收益率','roa_dp':'总资产净利率(杜邦分析)','debt_to_assets':'资产负债率','assets_turn':'总资产周转率','bps':'每股净资产','cfps':'每股现金流',
                      'ocf_to_opincome':'净现比','basic_eps_yoy':'每股收益同比增长率','cfps_yoy':'每股现金流同比增长率','or_yoy':'营业收入同比增长率','q_sales_yoy':'营业收入同比增长率(单季度)',
                      'q_profit_yoy':'净利润同比增长率(单季度)','equity_yoy':'净资产同比增长率','ar_turn':'应收账款周转率','ca_turn':'流动资产周转率','inv_turn':'存货周转率','undist_profit_ps':'每股未分配利润',
                      'tax_to_ebt':'所得税/利润总额'}
    income_dict = {'ts_code':'ts_code','end_date':'end_date','revenue':'营业收入','total_profit':'利润总额','income_tax':'所得税'}
    balance_dict = {'ts_code':'ts_code','end_date':'end_date','accounts_receiv':'应收账款','fix_assets':'固定资产','total_cur_liab':'流动负债'}
    # bz_profit = {'ts_code':'ts_code','end_date':'end_date','bz_profit':'主营业务利润'}
    # express_dict = {'ts_code':'ts_code','end_date':'end_date','total_assets':'总资产','open_net_assets':'期初净资产'}

    start_time = time.time()
    df_res = pd.DataFrame()

    for i,ts_code in enumerate(df_base['ts_code']):
        df_income = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date ,fields=','.join(income_dict.keys()))
        df_balance = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=','.join(balance_dict.keys()))
        df_indicator = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=','.join(indicator_dict.keys()))
        # df_express = pro.express(ts_code=ts_code, start_date=start_date, end_date=end_date,fields=','.join(express_dict.keys()))
        # df_bz = pro.fina_mainbz(ts_code=ts_code, start_date=start_date, end_date=end_date, type='P', fields=','.join(bz_profit.keys()))
        # df_stk = pro.stk_rewards(ts_code=ts_code)

        df_indicator.drop_duplicates(inplace=True)
        df_balance.drop_duplicates(inplace=True)
        df_income.drop_duplicates(inplace=True)

        # df_indicator.rename(columns=indicator_dict, inplace=True)
        # df_balance.rename(columns=balance_dict, inplace=True)
        # df_income.rename(columns=income_dict, inplace=True)

        df = pd.merge(df_indicator, df_balance, how='left', on=['ts_code','end_date'])
        df = pd.merge(df, df_income, how='left', on=['ts_code','end_date'])
        df_res = df_res.append(df, ignore_index=True)
        if i%10 == 0:
            print('finish',i,'/',len(df_base),'time', time.time()-start_time)
            time.sleep(12)
            start_time = time.time()
    df_res.to_sql('stock_season_basic', engine_ts, index=False, if_exists='replace' if replace else'append',
                  chunksize=5000)
    return df_res

def get_fund_basic():
    df_fund_base = pro.fund_basic(market='O', status='L')
    df_fund_base = df_fund_base[(df_fund_base['fund_type'] == '混合型') | (df_fund_base['fund_type'] == '股票型')]
    # res = df_fund_base.to_sql('fund_basic', engine_ts, index=False, if_exists='append', chunksize=5000)
    return df_fund_base

def get_season_fund(df_fund_base, start_date='20160101', end_date='20191231', replace=False):
    if end_date[-4:] == '1231': end_date = str(int(end_date[:4])+1)+'0401'
    elif end_date[-4:] == '0331': end_date = end_date[:4] + '0701'
    elif end_date[-4:] == '0630': end_date = end_date[:4] + '1001'
    elif end_date[-4:] == '0930': end_date = end_date[:4] + '1001'

    if start_date[-4:] == '0101': start_date = start_date[:4] + '0401'
    elif start_date[-4:] == '0401': start_date = start_date[:4] + '0701'
    elif start_date[-4:] == '0701': start_date = start_date[:4] + '1001'
    elif start_date[-4:] == '1001': start_date = str(int(start_date[:4])+1) + '0101'

    # start_date, end_date = '20160401', '20200401'

    set_dict = defaultdict(lambda : defaultdict(int))
    start_time = time.time()
    for i,ts_code in enumerate(df_fund_base['ts_code']):
        df = pro.fund_portfolio(ts_code=ts_code, start_date=start_date, end_date=end_date)
        g = df.groupby(['end_date'])
        for date, group in g:
            sorted_stock = group.sort_values(['stk_mkv_ratio'], ascending=False)
            sorted_stock.drop_duplicates(['symbol'], inplace=True)
            stock_lst = sorted_stock.head(10)['symbol'].tolist()
            for stock in stock_lst:
                set_dict[date][stock] += 1
        if i%10 == 0:
            print('finish',i,'/',len(df_fund_base),'time', time.time()-start_time)
            time.sleep(10)
            start_time = time.time()

    for i, (key, subdict) in enumerate(set_dict.items()):
        if key[-4:] in ['0331','0630','0930','1231']:
            df = pd.DataFrame({'symbol':list(subdict.keys()), 'count':list(subdict.values())})
            df.to_sql('holding_{}'.format(key), engine_ts, index=False, if_exists='replace' if replace else 'append', chunksize=5000)
            print(key,len(subdict))

def get_stock_monthly(df_base, start_date='20160101', end_date='20191231', replace=False):
    # start_date, end_date = '20160101', '20191231'
    monthly_dict = {'ts_code':'ts_code','trade_date':'交易日期','close':'月收盘价','open':'月开盘价','high':'月最高价','low':'月最低价',
                    'change':'月涨跌额','pct_chg':'月涨跌幅(复权)','vol':'月成交量','amount':'月成交额'}
    start_time = time.time()
    df_res = pd.DataFrame()
    for i, ts_code in enumerate(df_base['ts_code']):
        df_monthly = ts.pro_bar(ts_code=ts_code, adj='qfq', freq='M',start_date=start_date, end_date=end_date,adjfactor=True)
        if df_monthly is not None:
            # df_monthly.rename(columns=monthly_dict, inplace=True)
            df_res = df_res.append(df_monthly, ignore_index=True)
        if i % 10 == 0:
            print('finish', i, '/', len(df_base), 'time', time.time() - start_time)
            # time.sleep(5)
            start_time = time.time()

    df_res.to_sql('stock_monthly', engine_ts, index=False, if_exists='replace' if replace else 'append',
                      chunksize=5000)

def get_stock_daily(df_base, start_date='20160101', end_date='20191231', replace=False):
    # start_date, end_date = '20160101', '20191231'
    daily_dict = {'ts_code':'ts_code','trade_date':'交易日期','close':'当日收盘价','turnover_rate':'换手率',
                  'pe_ttm':'市盈率TTM','pb':'市净率','ps_ttm':'市销率（TTM）','dv_ttm':'股息率TTM',
                  'total_share':'总股本(万)','float_share':'流通股本(万)','total_mv':'总市值(万)','circ_mv':'流通市值(万)'}
    start_time = time.time()
    df_res = pd.DataFrame()
    for i, ts_code in enumerate(df_base['ts_code']):
        df_daily = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=','.join(daily_dict.keys()))
        # df_daily.rename(columns=daily_dict, inplace=True)
        df_res = df_res.append(df_daily,ignore_index=True)
        if i%10 == 0:
            print('finish',i,'/',len(df_base),'time', time.time()-start_time)
            time.sleep(5)
            start_time = time.time()
    df_res.to_sql('stock_daily', engine_ts, index=False, if_exists='replace'if replace else 'append', chunksize=5000)
    return df_res

def month2season(start_date='20160101', end_date='20191231', replace=False):
    df_stock_monthly = pd.read_sql('select * from stock_monthly', engine_ts)
    df_stock_monthly = df_stock_monthly[(df_stock_monthly['trade_date']>=start_date) & (df_stock_monthly['trade_date']<=end_date)]
    date_bin = [str(int(start_date[:4])-1)+'1231']
    for i in range(int(start_date[:4]),int(end_date[:4])+1):
        for last_4 in ['0331','0630', '0930', '1231']:
            date_bin.append(str(i)+last_4)
    if start_date == '0401': date_bin = date_bin[1:]
    elif start_date == '0701': date_bin = date_bin[2:]
    elif start_date == '1001': date_bin = date_bin[3:]

    if end_date == '0930': date_bin = date_bin[:-1]
    elif end_date == '0630': date_bin = date_bin[:-2]
    elif end_date == '0331': date_bin = date_bin[:-3]

    date_bin = [int(x) for x in date_bin]
    # date_bin = [20151231, 20160331, 20160630, 20160930, 20161231, 20170331, 20170630, 20170930,
    #             20171231, 20180331, 20180630, 20180930, 20181231, 20190331, 20190630, 20190930, 20191231]
    df_stock_monthly['trade_date'] = df_stock_monthly['trade_date'].astype(int)
    g = df_stock_monthly.groupby(['ts_code'])
    c = 0
    df_res = pd.DataFrame()
    start_time = time.time()
    for ts_code, group in g:
        group.sort_values(['trade_date'])
        date_groups = pd.cut(group['trade_date'], bins=date_bin)
        sub_g = group.groupby(date_groups)
        df1 = sub_g.agg({'high': np.max, 'low': np.min, 'vol': np.sum})
        df1.insert(0, 'ts_code', ts_code)
        df1['period'] = df1.index.astype(str)
        df1['period'] = df1['period'].apply(lambda x:x[-9:-1])
        for i in range(3):
            df1['pct_chg_{}'.format(i)] = np.nan
        for date, sub_group in sub_g:
            s = sub_group['pct_chg']
            for i, index in enumerate(s.index):
                df1.loc[date, 'pct_chg'.format(i)] = s[index]
        df_res = df_res.append(df1, ignore_index=True)
        c += 1
        if c%10==0:
            print('finish', c, 'time', time.time() - start_time)
            start_time = time.time()
    df_res.to_sql('stock_month_to_season', engine_ts, index=False, if_exists='replace'if replace else 'append',
               chunksize=5000)


def day2season(start_date='20160101', end_date='20191231', replace=False):
    df_stock_daily = pd.read_sql('select * from stock_daily', engine_ts)
    df_stock_daily = df_stock_daily[(df_stock_daily['trade_date']>=start_date) & (df_stock_daily['trade_date']<=end_date)]

    date_bin = [str(int(start_date[:4]) - 1) + '1231']
    for i in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for last_4 in ['0331', '0630', '0930', '1231']:
            date_bin.append(str(i) + last_4)
    if start_date == '0401':
        date_bin = date_bin[1:]
    elif start_date == '0701':
        date_bin = date_bin[2:]
    elif start_date == '1001':
        date_bin = date_bin[3:]

    if end_date == '0930':
        date_bin = date_bin[:-1]
    elif end_date == '0630':
        date_bin = date_bin[:-2]
    elif end_date == '0331':
        date_bin = date_bin[:-3]

    date_bin = [int(x) for x in date_bin]
    # date_bin = [20151231, 20160331, 20160630, 20160930, 20161231, 20170331, 20170630, 20170930,
    #             20171231, 20180331, 20180630, 20180930, 20181231, 20190331, 20190630, 20190930, 20191231]
    df_stock_daily['trade_date'] = df_stock_daily['trade_date'].astype(int)
    g = df_stock_daily.groupby(['ts_code'])
    c = 0
    df_res = pd.DataFrame()
    start_time = time.time()
    for ts_code, group in g:
        group.sort_values(['trade_date'])
        date_groups = pd.cut(group['trade_date'], bins=date_bin)
        sub_g = group.groupby(date_groups)
        df = sub_g.agg({'turnover_rate': np.mean, 'pe_ttm': [np.min, np.max, np.mean], 'pb': [np.min, np.max, np.mean],
                        'ps_ttm': [np.min, np.max, np.mean],'dv_ttm':np.mean,'total_share':np.mean,'total_mv':np.mean,
                        'float_share':np.mean, 'circ_mv':np.mean, 'close':np.std})
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df.insert(0, 'ts_code', ts_code)
        df['period'] = df.index.astype(str)
        df['period'] = df['period'].apply(lambda x: x[-9:-1])

        df_res = df_res.append(df, ignore_index=True)
        c += 1
        if c % 10==0:
            print('finish', c, 'time', time.time() - start_time)
            start_time = time.time()
    df_res.to_sql('stock_day_to_season', engine_ts, index=False, if_exists='replace' if replace else 'append', chunksize=5000)


def visualize(date_lst= ['20161231']):
    date_lst = ['20161231','20171231','20181231','20191231']
    g_lst = []
    for i, date in enumerate(date_lst):
        train_set = pd.read_sql('select * from holding_{}'.format(date), engine_ts)
        count_bin = [0,1,3,5]+list(range(10,110,10))+ list(range(150,550,50)) + [1000,10000]

        count_groups = pd.cut(train_set['count'], bins=count_bin)
        g = train_set.groupby(count_groups)['symbol'].count()
        g.name = date
        g_lst.append(g)

    df_count = pd.concat(g_lst, axis=1)
    # plt.figure()
    df_count.plot(kind='bar')
    plt.show()

def visualize_holding_number():
    date_lst = [20160331, 20160630, 20160930, 20161231, 20170331, 20170630, 20170930,
     20171231, 20180331, 20180630, 20180930, 20181231, 20190331, 20190630, 20190930, 20191231]
    count_bin = [0,5,20,50,100,500,3000]
    date_lst = [str(x) for x in date_lst]
    res = []
    for i, date in enumerate(date_lst):
        all_set = pd.read_sql('select * from holding_{}'.format(date), engine_ts)
        count_groups = pd.cut(all_set['count'], bins=count_bin)
        count_num = all_set.groupby(count_groups)['count'].count()
        count_num = count_num.to_frame()
        count_num.rename(columns={'count':date}, inplace=True)
        res.append(count_num)

    res = pd.concat(res, 1)
    # res = res.stack().unstack(0)
    res = res.T
    res.to_csv('visualize_count.csv')

    res = pd.read_csv('visualize_count.csv')
    final = pd.DataFrame()
    final['end_date'] = res[res.columns[0]]
    for i in range(len(count_bin)-1):
        final['{}'.format(count_bin[i+1])] = res[res.columns[i+1]]
    # res.rename(columns={'Unnamed: 0':'end_date'}, inplace=True)
    final.to_sql('visualize_count', engine_ts, index=False, if_exists='replace', chunksize=5000)



def get_label(truncate=3, start_date='20160101', end_date='20191231', replace=False):
    date_lst = []
    for i in range(int(start_date[:4]),int(end_date[:4])+1):
        for last_4 in ['0331','0630', '0930', '1231']:
            date_lst.append(str(i)+last_4)
    if start_date == '0401': date_lst = date_lst[1:]
    elif start_date == '0701': date_lst = date_lst[2:]
    elif start_date == '1001': date_lst = date_lst[3:]

    if end_date == '0930': date_lst = date_lst[:-1]
    elif end_date == '0630': date_lst = date_lst[:-2]
    elif end_date == '0331': date_lst = date_lst[:-3]
    # date_lst = [20160331, 20160630, 20160930, 20161231, 20170331, 20170630, 20170930,
    #  20171231, 20180331, 20180630, 20180930, 20181231, 20190331, 20190630, 20190930, 20191231]
    # date_lst = [str(x) for x in date_lst]
    pos_lst = []

    for i, date in enumerate(date_lst):
        all_set = pd.read_sql('select * from holding_{}'.format(date), engine_ts)
        pos_set = all_set[all_set['count']>=truncate]
        pos_set['end_date'] = date
        pos_set['label'] = 1
        pos_set.rename(columns={'symbol':'ts_code'}, inplace=True)
        pos_set.drop(columns='count',inplace=True)
        pos_lst.append(pos_set)

    df_pos = pd.concat(pos_lst, axis=0)

    df_stock_season = pd.read_sql('select * from stock_season_basic', engine_ts)
    df = pd.merge(df_stock_season, df_pos, how='left', on=['ts_code','end_date'])

    df_stock_day2season = pd.read_sql('select * from stock_daily_to_season', engine_ts)
    df_stock_day2season.rename(columns={'period':'end_date'}, inplace=True)
    df = pd.merge(df, df_stock_day2season, how='left', on=['ts_code', 'end_date'])

    df_stock_month2season = pd.read_sql('select * from stock_month_to_season', engine_ts)
    df_stock_month2season.rename(columns={'period':'end_date'}, inplace=True)
    df = pd.merge(df, df_stock_month2season, how='left', on=['ts_code', 'end_date'])

    df_stock_basic = pd.read_sql('select * from stock_basic', engine_ts)
    df = pd.merge(df, df_stock_basic, how='left', on=['ts_code'])

    df.drop_duplicates(['ts_code','end_date'], inplace=True)
    df['label'].fillna(0, inplace=True)

    df['end_date'] = df['end_date'].astype(int)
    stock_group = df.groupby('ts_code')
    c = 0
    df_res = pd.DataFrame()
    for ts_code, group in stock_group:
        c += 1
        group.sort_values(['end_date'], ascending=True, inplace=True)
        group['label_new'] = 0
        for i in range(len(group)-1):
            if group.iloc[i+1]['label'] - group.iloc[i]['label'] == 1:
                group.loc[group.index[i],'label_new'] = 1
        df_res = df_res.append(group, ignore_index=True)
        if c % 10 ==0:
            print(c)

    df_res.to_sql('train_data_{}'.format(truncate), engine_ts, index=False, if_exists='replace'if replace else 'append', chunksize=5000)


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def corr_heatmap(truncate=3):
    df_data = pd.read_sql('select * from train_data_fillna_{}'.format(truncate), engine_ts)
    feature_lst = list(set(final_dict.keys()) - set(['symbol','ts_code','end_date','symbol','name','area','industry','market','list_date','setup_date','close_std', 'chg_max_pos', 'chg_max_neg']))
    df_data = df_data[feature_lst]
    correlations = df_data.corr()
    correction=abs(correlations)
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(correction, vmax=1, square=True)
    plt.savefig('Correlation-Matrix.png')
    for each in feature_lst:
        print(each, final_dict[each])
    correlations = df_data.corr()['label_new'].sort_values()
    # Display correlations
    print('Most Positive Correlations:\n', correlations.tail(30))
    print('\nMost Negative Correlations:\n', correlations.head(30))

    # for each in feature_lst:
    #     if each != 'label_new':
    #         f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,4))
    #         bins = 30
    #         ax1.hist(df_data[df_data["label_new"]== 1][each], bins = bins)
    #         ax1.set_title('Fraud')
    #
    #         ax2.hist(df_data[df_data["label_new"] == 0][each], bins = bins)
    #         ax2.set_title('Normal')
    #
    #         plt.xlabel(each)
    #         plt.ylabel('Number of being held')
    #         # plt.yscale('log')
    #         if each == 'float_share/total_share':
    #             each = 'float_share_chu_total_share'
    #         plt.savefig('image/{}.png'.format(each))


def corr_analy(truncate=3, replace=False):
    df_data = pd.read_sql('select * from train_data_{}'.format(truncate), engine_ts)
    #类型转换
    convert_lst = ['current_ratio', 'quick_ratio', 'inv_turn', 'ar_turn', 'ca_turn', 'total_cur_liab']
    for each in convert_lst:
        df_data[each] = df_data[each].astype(float)


    # df_data['流动比率'] = df_data['流动比率'].astype(float)
    # df_data['速动比率'] = df_data['速动比率'].astype(float)
    # df_data['存货周转率'] = df_data['存货周转率'].astype(float)
    # df_data['应收账款周转率'] = df_data['应收账款周转率'].astype(float)
    # df_data['流动资产周转率'] = df_data['流动资产周转率'].astype(float)
    # df_data['流动负债'] = df_data['流动负债'].astype(float)

    #计算指标
    # df_data['流通A比例'] = df_data['流通股本(万)_mean']/df_data['总股本(万)_mean']
    # df_data['上市时间'] = df_data['end_date'].apply(lambda x:int(str(x)[:4]))-df_data['上市日期'].apply(lambda x:int(str(x)[:4]))
    # df_data['公司成立时间'] = df_data['end_date'].apply(lambda x:int(str(x)[:4]))-df_data['注册日期'].apply(lambda x:int(str(x)[:4]))
    df_data['list_date'].fillna('null', inplace=True)
    df_data.drop(df_data[df_data['list_date'] =='null'].index, inplace=True)
    df_data['float_share_to_total_share'] = df_data['float_share_mean']/df_data['total_share_mean']
    df_data['list_time'] = df_data['end_date'].apply(lambda x:int(str(x)[:4]))-df_data['list_date'].apply(lambda x:int(str(x)[:4]))
    df_data['setup_date'] = df_data['end_date'].apply(lambda x:int(str(x)[:4]))-df_data['setup_date'].apply(lambda x:int(str(x)[:4]))
    df_data['chg_max_pos'] = (df_data['high'] - df_data['low'])/df_data['low'] if df_data['low'] is not None else 0
    df_data['chg_max_neg'] = (df_data['high'] - df_data['low'])/df_data['high'] if df_data['high'] is not None else 0

    print(df_data.columns)
    print(len(df_data))
    df_data['filter'] = df_data['end_date'].apply(lambda x: str(x)[-4:])
    df_data.drop(df_data[df_data['filter'] == '1231'].index, inplace=True)
    print(len(df_data))

    # feature_lst = ['每股收益', '每股营业收入', '每股资本公积', '每股盈余公积',
    #                '每股未分配利润', '流动比率', '速动比率', '存货周转率', '应收账款周转率', '流动资产周转率', '总资产周转率',
    #                '每股净资产', '每股现金流', '净利润/营业总收入', '营业利润/营业总收入', '净资产收益率', '加权平均净资产收益率',
    #                '所得税/利润总额', '净现比', '资产负债率', '总资产净利率(杜邦分析)', '每股收益(单季度)',
    #                '净利润/营业总收入(单季度)', '营业利润/营业总收入(单季度)', '净资产收益率(单季度)', '每股收益同比增长率',
    #                '每股现金流同比增长率', '营业收入同比增长率', '营业收入同比增长率(单季度)', '净利润同比增长率(单季度)',
    #                '净资产同比增长率', '应收账款', '固定资产', '流动负债', '营业收入', '利润总额', '所得税', 'label',
    #                '换手率_mean', '市盈率TTM_amin', '市盈率TTM_amax', '市盈率TTM_mean', '市净率_amin',
    #                '市净率_amax', '市净率_mean', '市销率（TTM）_amin', '市销率（TTM）_amax',
    #                '市销率（TTM）_mean', '股息率TTM_mean', '总股本(万)_mean', '总市值(万)_mean',
    #                '流通股本(万)_mean', '流通市值(万)_mean', '季最高价', '季最低价', '季成交量', '月涨跌幅(复权)_0',
    #                '月涨跌幅(复权)_1', '月涨跌幅(复权)_2',
    #                '员工人数', '流通A比例', '上市时间', '公司成立时间','季最大涨幅','季最大跌幅']
    feature_lst = list(set(final_dict.keys()) - set(['symbol','end_date','ts_code','ann_date','name','area','industry','market','list_date','setup_date']))

    # final_dict_reversed = {v:k for (k,v) in final_dict.items()}

    contrast = pd.DataFrame(columns=('feature','mean_1','std_1', 'len_1','t_1','mean_0', 'std_0', 'len_0','t_0'))
    for key in feature_lst:
        # if key in final_dict_reversed:
        #     each = final_dict_reversed[key]
        # else:
        each = key
        mean_1 = df_data[df_data['label_new']==1][each].mean()
        mean_0 = df_data[df_data['label_new']==0][each].mean()
        std_1 = df_data[df_data['label_new'] == 1][each].std()
        std_0 = df_data[df_data['label_new'] == 0][each].std()
        len_1 = len(df_data[df_data['label_new']==1][each])
        len_0 = len(df_data[df_data['label_new']==0][each])
        contrast = contrast.append([{'feature':key,'mean_1':mean_1,'std_1':std_1,'len_1':len_1,'mean_0':mean_0, 'std_0':std_0,'len_0':len_0}], ignore_index=True)
    contrast.to_sql('contrast_{}'.format(truncate), engine_ts, index=False, if_exists='replace', chunksize=5000)

    #缺失值填充
    df_data['pb_amin'].fillna(0, inplace=True)
    df_data['pb_amax'].fillna(0, inplace=True)
    df_data['pb_mean'].fillna(0, inplace=True)
    df_data['dv_ttm_mean'].fillna(0, inplace=True)

    stock_group = df_data.groupby('ts_code')
    group_lst = []
    for ts_code, group in stock_group:
        group.fillna(group.mean(), inplace=True)
        group_lst.append(group)
    df_data = pd.concat(group_lst)

    df_data.fillna(df_data.mean(), inplace=True)

    df_data.to_sql('train_data_fillna_{}'.format(truncate), engine_ts, index=False, if_exists='replace'if replace else 'append', chunksize=5000)

    print(df_data.dtypes.value_counts())
    print('-------------------------------------------------')

    # Number of unique classes in each object column
    print(df_data.select_dtypes('object').apply(pd.Series.nunique, axis=0))
    print('-------------------------------------------------')

    # Missing values statistics
    missing_values = missing_values_table(df_data)
    print(missing_values)
    print('-------------------------------------------------')

    print(df_data['label_new'].value_counts())
    print('-------------------------------------------------')

    correlations = df_data.corr()['label_new'].sort_values()
    # Display correlations
    print('Most Positive Correlations:\n', correlations.tail(30))
    print('\nMost Negative Correlations:\n', correlations.head(30))

def get_ar(df_base, start_date='20160101', end_date='20191231', replace=False):

    start_time = time.time()
    df_res = pd.DataFrame()
    for i, ts_code in enumerate(df_base['ts_code']):
        df_stock_car = ts.pro_bar(ts_code=ts_code, adj='qfq', freq='D', start_date=start_date, end_date=end_date)
        if df_stock_car is not None:
            df_stock_car = df_stock_car[['ts_code', 'trade_date', 'pct_chg']]
            df_index = ts.pro_bar(ts_code='000300.SH', adj='qfq', asset='I', freq='D', start_date=start_date, end_date=end_date)
            df_index = df_index[['trade_date', 'pct_chg']]
            df_index.rename(columns={'pct_chg': 'pct_chg_index'}, inplace=True)
            df_stock_car = pd.merge(df_stock_car, df_index, how='left', on=['trade_date'])
            df_stock_car['ar'] = df_stock_car['pct_chg'] - df_stock_car['pct_chg_index']
            df_res = df_res.append(df_stock_car, ignore_index=True)
        if i%10 == 0:
            print('finish',i,'/',len(df_base),'time', time.time()-start_time)
            start_time = time.time()
    df_res.to_sql('stock_car', engine_ts, index=False, if_exists='replace' if replace else 'append', chunksize=5000)

    return

def get_car(start_date='20160101', end_date='20191231'):
    df_ar = pd.read_sql('select * from stock_car', engine_ts)

    date_bin = [str(int(start_date[:4]) - 1) + '1231']
    for i in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for last_4 in ['0331', '0630', '0930', '1231']:
            date_bin.append(str(i) + last_4)
    if start_date == '0401':
        date_bin = date_bin[1:]
    elif start_date == '0701':
        date_bin = date_bin[2:]
    elif start_date == '1001':
        date_bin = date_bin[3:]

    if end_date == '0930':
        date_bin = date_bin[:-1]
    elif end_date == '0630':
        date_bin = date_bin[:-2]
    elif end_date == '0331':
        date_bin = date_bin[:-3]

    # date_groups = pd.cut(df_ar['trade_date'], bins=date_bin)
    # sub_g = group.groupby(date_groups)



def end_to_trade(end_date):
    if end_date[-4:] == '0331':
        trade_date = [end_date[:4]+'0101',end_date[:4]+'0630']
    elif end_date[-4:] == '0630':
        trade_date = [end_date[:4]+'0401',end_date[:4]+'0930']
    elif end_date[-4:] == '0930':
        trade_date = [end_date[:4]+'0701',end_date[:4]+'1231']
    elif end_date[-4:] == '1231':
        trade_date = [end_date[:4]+'1001',str(int(end_date[:4])+1)+'0331']
    else:
        print('false end_date input')
        return
    return trade_date[0], trade_date[1]

def visualize_car(truncate=3, end_date = '20190331'):
    df = pd.read_sql('select * from train_data_{}'.format(truncate), engine_ts)[['ts_code', 'end_date', 'label_new']]
    df['end_date'] = df['end_date'].astype(str)
    df_car = pd.read_sql('select * from stock_car', engine_ts)[['ts_code', 'trade_date', 'ar']]

    df_1 = df[(df['label_new']==1) & (df['end_date'] == end_date)]
    df_0 = df[(df['label_new']==0) & (df['end_date'] == end_date)]

    df_res = pd.DataFrame()
    for label, df_label in enumerate([df_1, df_0]):
        for i, (_, row) in enumerate(df_label.iterrows()):
            trade_date_min, trade_date_max = end_to_trade(row['end_date'])
            df_row = df_car[(df_car['ts_code']==row['ts_code']) & (df_car['trade_date'] >= trade_date_min) & (df_car['trade_date'] <= trade_date_max)]
            df_row.sort_values(['trade_date'], ascending=True, inplace=True)
            df_row['end_date'] = row['end_date']
            df_res = df_res.append(df_row, ignore_index=True)
            print(i, '/', len(df_label))

        df_res = df_res.groupby('trade_date').mean()
        df_res.sort_values(['trade_date'], inplace=True, ascending=True)
        df_res = df_res.reset_index()
        df_res['car'] = df_res['ar'].cumsum()
        # df_res.index = df_res.index - df_res[df_res['trade_date'] == end_date].index[0]
        ax = df_res['car'].plot()
        fig = ax.get_figure()
        fig.savefig('{}_car_{}.png'.format(end_date, 1-label))


def show_result(truncate=3):
    df = pd.read_sql('select * from train_data_fillna_{}'.format(truncate), engine_ts)
    df = df[['ts_code', 'end_date','symbol', 'name', 'total_mv_mean', 'float_share_to_total_share','eps','label_new']]

    df.to_sql('display_prediction', engine_ts, index=False, if_exists='replace', chunksize=5000)




if __name__ == '__main__':
    engine_ts = create_engine('mysql+pymysql://test:123456@47.103.137.116:3306/testDB?charset=utf8&use_unicode=1')
    pro = ts.pro_api('4c20c75b7e45fa73eefd12cf0eac8b8b89bd801215d910a2965d62cf')

    ##爬取数据
    # df_base = get_basic_info() #股票基本信息
    df_base = pro.stock_basic()
    #
    # print(1)
    # df_stock_season = get_season_basic(df_base=df_base, start_date='20200101', end_date='20200630', replace=False) #股票财务信息
    # print(2)
    # df_stock_monthly = get_stock_monthly(df_base=df_base,start_date='20200101', end_date='20200630', replace=False)  #股票月行情
    # print(3)
    # df_stock_daily = get_stock_daily(df_base=df_base, start_date='20200101', end_date='20200630', replace=False)  #股票日行情
    # print(4)
    # df_fund_base = get_fund_basic()  #基金基本信息
    # get_season_fund(df_fund_base, start_date='20200101', end_date='20200630', replace=False)  #基金季度持仓


    # 整理为基金季度特征
    # month2season() #月线
    # day2season(start_date='20160101', end_date='20191231', replace=True) #日线

    # visualize()
    # revise_name()
    # truncate=3
    # get_label(truncate=truncate, replace=True)
    # corr_analy(truncate=truncate, replace=True)
    # corr_heatmap()


    # df_base = pro.stock_basic()
    # get_car(df_base, start_date='20200101', end_date='20200331')


    # end_date = '20191231'
    # print(end_date)
    # visualize_car(truncate=truncate, end_date = end_date)
    # visualize_holding_number()
    show_result(3)



















