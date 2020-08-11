import pandas as pd
from sqlalchemy import create_engine
import json
import numpy as np
import os


def get_train_data(truncate, train_year=['2016', '2017', '2018'], test_season=['20190331', '20190630', '20190930'], read_sql=True):
    engine_ts = create_engine(
        'mysql+pymysql://test:123456@47.103.137.116:3306/testDB?charset=utf8&use_unicode=1')
    if read_sql:
        try:
            df = pd.read_sql(
                'select * from train_data_fillna_{}'.format(truncate), engine_ts)
            print('query success')
        except BaseException as e:
            print(e)
    else:
        df = pd.read_csv('data/train_data_fillna_3.csv', index_col=False)

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
    # df_train.to_csv('train_data.csv')

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
    df = pd.read_csv('data/display_prediction.csv', index_col=False)
    # df = pd.read_sql('select * from display_prediction', engine_ts)
    df['end_date'] = df['end_date'].astype(str)

    test_name['probability'] = y_pred
    res = pd.merge(test_name, df, how='left', on=['ts_code', 'end_date'])
    res.sort_values('probability', ascending=False, inplace=True)
    # res.to_sql('result_{}'.format(test_season[0]), engine_ts, index=False,
    #                  if_exists='replace', chunksize=5000)
    return res

def get_predicted_and_real(res, num=30):
    engine_ts = create_engine('mysql+pymysql://test:123456@47.103.137.116:3306/testDB?charset=utf8&use_unicode=1')

    res = res.reset_index(drop=True)
    res_predicted = res.iloc[:num][['ts_code','name','label_new','probability']]
    res_predicted = res_predicted.sort_values(['label_new','probability'], ascending=False)[['ts_code','name','label_new']]
    res_predicted.rename(columns={'ts_code':'ts_code_predicted','name':'name_predicted'}, inplace=True)
    res_predicted = res_predicted.reset_index(drop=True)
    res_real = res[res['label_new']==1].iloc[:num][['ts_code','name']]
    res_real.rename(columns={'ts_code':'ts_code_real','name':'name_real'}, inplace=True)
    res_real = res_real.reset_index(drop=True)
    predicted_and_real = pd.concat([res_predicted,res_real], axis=1)
    # predicted_and_real.to_sql('predicted_and_real_{}', engine_ts, index=False, if_exists='replace', chunksize=5000)
    return predicted_and_real


def get_industry(res, num=50):
    df_industry = pd.read_csv('data/stock_industry.csv', index_col=False)
    res_predicted = res.iloc[:num][['ts_code', 'name', 'probability']]
    res_predicted = res_predicted.sort_values(['probability'], ascending=False)[['ts_code', 'name']]
    res_real = res[res['label_new'] == 1][['ts_code', 'name', 'probability']]

    res_predicted = pd.merge(res_predicted, df_industry, how='left', on=['ts_code'])
    count_predicted = res_predicted['industry'].value_counts()
    count_predicted = count_predicted[count_predicted.values>1]
    res_real = pd.merge(res_real, df_industry, how='left', on=['ts_code'])
    count_real = res_real['industry'].value_counts()
    count_real = count_real[count_real.values>5]

    return count_predicted.to_dict(), count_real.to_dict()


    # import matplotlib.pyplot as plt
    # labels = list(count_predicted.index)
    # X = count_predicted.values
    # plt.figure()
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # plt.title("Pie chart")
    # plt.show()
    # plt.close()
    # labels = list(count_real.index)
    # X = count_real.values
    # plt.figure()
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # plt.show()

