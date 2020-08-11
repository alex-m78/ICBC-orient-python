import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score
from data_preprocessing import get_train_data, output_result, get_predicted_and_real
from utils import precision_n, recall_n, precision_50
from sqlalchemy import create_engine
from sklearn.metrics import make_scorer
import pickle
import time
import os

def get_xgb_prediction(test_season=['20180930'], load=False, read_sql=True):

    time1 = time.time()
    truncate = 3
    x_train, x_test, y_train, y_test, test_name = get_train_data(truncate=truncate,
                                                                 train_year=['2016', '2017', '2018', '2019'],
                                                                 test_season=test_season, read_sql=read_sql)
    print('load data time:',time.time()-time1)

    other_params = {'eta': 0.3, 'n_estimators': 120, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 12,
                    'learning_rate': 0.1,'colsample_bytree': 1, 'colsample_bylevel': 0.5, 'subsample': 1.0, 'reg_lambda': 40,
                    'reg_alpha': 10,'seed': 2020, 'scale_pos_weight': 1}
    time2 = time.time()
    if load:
        model = pickle.load(open(os.getcwd()+"/saved_model/xgb_{}.dat".format(test_season[0]), "rb"))
    else:
        model = xgb.XGBClassifier(**other_params)
        model.fit(x_train, y_train)
        pickle.dump(model, open(os.getcwd()+ "/saved_model/xgb_{}.dat".format(test_season[0]), "wb"))
    print('computed time:',time.time()-time2)
    time3 = time.time()

    y_pred = model.predict_proba(x_test)[:, 1]

    res = output_result(y_pred, test_name, test_season)
    predicted_and_real = get_predicted_and_real(res)
    print('p10:{:.4f},p30:{:.4f},p50:{:.4f},p100:{:.4f}'.format(precision_n(y_test, y_pred, 10),precision_n(y_test, y_pred, 30),precision_n(y_test, y_pred, 50),precision_n(y_test, y_pred, 100)))
    acc = accuracy_score(y_test, y_pred>0.3)
    print('accuracy:', acc)
    print('output time:',time.time()-time3)
    return res, predicted_and_real, acc, precision_n(y_test, y_pred, 30),

def xgb_tuning(train_year=['2016', '2017', '2018', '2019'], test_season=['20180930']):
    truncate = 3
    x_train, x_test, y_train, y_test, test_name = get_train_data(truncate=truncate,
                                                                 train_year=train_year,
                                                                 test_season=test_season)
    other_params = {'eta': 0.3, 'n_estimators': 120, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 12, 'learning_rate': 0.1,
                    'colsample_bytree': 1, 'colsample_bylevel': 0.5, 'subsample': 1.0, 'reg_lambda': 40, 'reg_alpha': 10,
                    'seed': 1302, 'scale_pos_weight': 1}
    model = xgb.XGBClassifier(**other_params)

    #=================================================================

    # cv_params = {'n_estimators': np.linspace(30, 150, 13, dtype=int)}
    cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}
    cv_params = {'min_child_weight': np.linspace(5, 15, 11, dtype=int)}
    cv_params = {'gamma': np.linspace(0, 1, 5)}
    cv_params = {'subsample': np.linspace(0, 1, 6)}
    cv_params = {'reg_lambda': np.linspace(0, 100, 11)}
    cv_params = {'reg_alpha': np.linspace(0, 10, 11)}
    cv_params = {'scale_pos_weight': np.linspace(0, 10, 11)}

    p50_score = make_scorer(precision_50, greater_is_better=True, needs_proba = True)
    gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=10, scoring=p50_score)
    gs.fit(x_train, y_train)
    print("参数的最佳取值：:", gs.best_params_)
    print("最佳模型得分:", gs.best_score_)


# idx = np.argsort(-model.feature_importances_)
# print(x_train.columns[idx])
# print(model.feature_importances_[idx])



