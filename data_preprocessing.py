import pandas as pd
from sqlalchemy import create_engine
import json
import numpy as np
import os


def get_train_data(truncate, train_year=['2016', '2017', '2018'], test_season=['20190331', '20190630', '20190930']):
    engine_ts = create_engine(
        'mysql+pymysql://test:123456@47.103.137.116:3306/testDB?charset=utf8&use_unicode=1')

    try:
        df = pd.read_sql(
            'select * from train_data_fillna_{}'.format(truncate), engine_ts)
        print('query success')
    except BaseException as e:
        print(e)

    convert_dict = json.load(
        open(os.getcwd() + '/convert_dict.json', 'r', encoding='utf-8'))
    convert_dict_reversed = {v: k for (k, v) in convert_dict.items()}

    base_dict = {'ts_code': 'ts_code', 'end_date': 'end_date'}

    onehot_lst = ['market']
    onehot_dict = {'创业板': 'cyb', '主板': 'zb', '中小板': 'zxb', '科创板': 'kcb'}

    convert_dict_reversed['label'] = 'label'
    convert_dict_reversed = {
        **convert_dict_reversed, **onehot_dict, **base_dict}

    onehot_frame = []
    for each in onehot_lst:
        tmp = pd.get_dummies(df[each])
        onehot_frame.append(tmp)
    onehot_train = pd.concat(onehot_frame, axis=1)
    onehot_train.rename(columns=onehot_dict, inplace=True)
    # =====================================================================

    select_feature = list(set(convert_dict_reversed.values(
    )) - set(['symbol', 'ann_date', 'name', 'area', 'industry', 'market', 'list_date', 'setup_date']))
    select_feature.sort()

    df_train = pd.concat([df, onehot_train], 1)[select_feature]

    x_train, x_test, y_train, y_test, test_name = split_data(
        df_train, train_year=train_year, test_season=test_season)

    return x_train, x_test, y_train, y_test, test_name


def split_data(df_train, train_year, test_season):
    df_train['end_date'] = df_train['end_date'].astype(str)

    train, test = pd.DataFrame(), pd.DataFrame()

    g = df_train.groupby(['end_date'])
    for date, group in g:
        if date[:4] in train_year and date not in test_season:
            train = train.append(group, ignore_index=True)
        elif date in test_season:
            test = test.append(group, ignore_index=True)

    train = train.sample(frac=1.0, random_state=2020, axis=0)
    y_train = train['label_new']
    x_train = train.drop(columns=['label_new', 'ts_code', 'end_date'])
    y_test = test['label_new']
    test_name = test[['ts_code', 'end_date']]
    x_test = test.drop(columns=['label_new', 'ts_code', 'end_date'])

    x_train, x_test = x_train.astype('float'), x_test.astype('float')

    return x_train, x_test, y_train, y_test, test_name


def end_to_trade(end_date):
    if end_date[-4:] == '0331':
        trade_date = [end_date[:4]+'0401', end_date[:4]+'0630']
    elif end_date[-4:] == '0630':
        trade_date = [end_date[:4]+'0701', end_date[:4]+'0930']
    elif end_date[-4:] == '0930':
        trade_date = [end_date[:4]+'1001', end_date[:4]+'1231']
    elif end_date[-4:] == '1231':
        trade_date = [str(int(end_date[:4])+1)+'0101',
                      str(int(end_date[:4])+1)+'0331']
    else:
        print('false end_date input')
        return
    return trade_date[0], trade_date[1]


def output_result(y_pred, test_name, test_season):
    engine_ts = create_engine(
        'mysql+pymysql://test:123456@47.103.137.116:3306/testDB?charset=utf8&use_unicode=1')
    df = pd.read_sql('select * from display_prediction', engine_ts)
    df['end_date'] = df['end_date'].astype(str)

    test_name['probability'] = y_pred
    test_name.to_sql('temp_result', engine_ts, index=False,
                     if_exists='replace', chunksize=5000)
    res = pd.merge(test_name, df, how='left', on=['ts_code', 'end_date'])

    res.sort_values('probability', ascending=False, inplace=True)
    res = res

    # res.to_csv('result/res_{}.csv'.format(test_season[0]))
    return res

# g = res.groupby('end_date')
# for end_date, sub_g in g:
#     sub_g.sort_values('probability', ascending=False, inplace=True)
#     sub_g = sub_g.iloc[:100]
#
#     car = []
#     for i, (_, row) in enumerate(sub_g.iterrows()):
#         trade_date_min, trade_date_max = end_to_trade(row['end_date'])
#         df_row = df_car[(df_car['ts_code'] == row['ts_code']) & (df_car['trade_date'] >= trade_date_min) & (
#                     df_car['trade_date'] <= trade_date_max)]
#         car.append(df_row['ar'].sum())
#
#     sub_g['car'] = car
#     print(sub_g.iloc[:20])
#     print('car: top10:{:.4f}, top20:{:.4f}, top50:{:.4f}, top100:{:.4f}'
#           .format(sub_g.iloc[:10]['car'].sum(),sub_g.iloc[:20]['car'].sum(), sub_g.iloc[:50]['car'].sum(),sub_g.iloc[:100]['car'].sum()))
#     sub_g.to_csv('../result/res_{}.csv'.format(end_date))


# car = []
# for i, (_, row) in enumerate(res.iterrows()):
#     trade_date_min, trade_date_max = end_to_trade(row['end_date'])
#     df_row = df_car[(df_car['ts_code'] == row['ts_code']) & (df_car['trade_date'] >= trade_date_min) & (
#                 df_car['trade_date'] <= trade_date_max)]
#     car.append(df_row['ar'].sum())
#
# res['car'] = car

# print(res.iloc[:20])
# print('car: top10:{:.4f}, top20:{:.4f}, top50:{:.4f}, top100:{:.4f}'
#       .format(res.iloc[:10]['car'].sum(),res.iloc[:20]['car'].sum(), res.iloc[:50]['car'].sum(),res.iloc[:100]['car'].sum()))
