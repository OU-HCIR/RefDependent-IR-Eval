import numpy as np
import warnings
warnings.filterwarnings("ignore")
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, mean_squared_error, mean_absolute_error
import os

def calc_dcg(r, rank_col, score_col, at):
    '''
    rank_col: 当前页面排序列名
    score_col: relevant score列名
    at: int, calculate dcg@at
    '''
    sorted_vec = list(zip(r[rank_col], r[score_col]))

    ranking = [t[1] for t in sorted_vec[0: at]]
    dcg_ = np.sum([(2 ** r - 1) / math.log(i + 2, 2) for i, r in enumerate(ranking)])
    return dcg_


def calc_ndcg(r, rank_col, score_col, at):
    '''
    rank_col: 当前页面排序列名
    score_col: relevant score列名
    at: int, calculate dcg@at
    '''
    sorted_vec = r.sort_values([score_col], ascending=False).reset_index(drop=True)
    ideal_dcg = calc_dcg(sorted_vec, rank_col, score_col, at)

    sorted_vec = r.sort_values([rank_col], ascending=True).reset_index(drop=True)
    cur_dcg = calc_dcg(sorted_vec, rank_col, score_col, at)
    if ideal_dcg == 0:
        return 0
    else:
        return cur_dcg / ideal_dcg

"""
================================================================================================
features at time t-1 minus features at time t-2.
================================================================================================
"""
def first_reference_t_1(r, data, col):
    if r['QueryOrder'] <= 2:
        return
    else:
        first_df = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == 1), col]
        last_df = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        return last_df.values[0] - first_df.values[0]

def peak_reference_t_1(r, data, col):
    if r['QueryOrder'] <= 2:
        return
    else:
        before_df_t_2 = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder'] - 1), col]
        before_df_t_1 = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder'] - 1), col]
        return before_df_t_1.values[0] - before_df_t_2.max()

def ave_reference_t_1(r, data, col):
    if r['QueryOrder'] <= 2:
        return
    else:
        before_df_t_2 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder']-1), col]
        before_df_t_1 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        return before_df_t_1.values[0]-before_df_t_2[before_df_t_2.isnull()==False].mean()

def end_reference_t_1(r, data, col):
    if r['QueryOrder'] <= 2:
        return
    else:
        before_df_t_2 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-2), col]
        before_df_t_1 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        return before_df_t_1.values[0]-before_df_t_2.values[0]

def peak_end_reference_t_1(r, data, col):
    if r['QueryOrder'] <= 2:
        return
    else:
        before_df_t_2_end = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-2), col]
        before_df_t_2_max = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] <= r['QueryOrder']-2), col].max()
        before_df_t_1 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        return before_df_t_1.values[0]-(before_df_t_2_end.values[0]+before_df_t_2_max)/2
"""
============================================================================================
features at time t minus features at time t-1
============================================================================================
"""
def first_reference_t(r, data, col):
    if r['QueryOrder'] <= 1:
        return
    else:
        first_df = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == 1), col]
        last_df = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']), col]
        return last_df.values[0] - first_df.values[0]

def peak_reference_t(r, data, col):
    if r['QueryOrder'] <= 1:
        return
    else:
        before_df_t_1 = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder']), col]
        before_df_t = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']), col]
        return before_df_t.values[0] - before_df_t_1.max()

def ave_reference_t(r, data, col):
    if r['QueryOrder'] <= 1:
        return
    else:
        before_df_t_1 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder']), col]
        before_df_t = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']), col]
        return before_df_t.values[0]-before_df_t_1[before_df_t_1.isnull() == False].mean()

def end_reference_t(r, data, col):
    if r['QueryOrder'] <= 1:
        return
    else:
        before_df_t_1 = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        before_df_t = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']), col]
        return before_df_t.values[0]-before_df_t_1.values[0]

def peak_end_reference_t(r, data, col):
    if r['QueryOrder'] <= 1:
        return
    else:
        before_df_t_1_end = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']-1), col]
        before_df_t_1_max = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] <= r['QueryOrder']-1), col].max()
        before_df_t = data.loc[(data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder']), col]
        return before_df_t.values[0]-(before_df_t_1_end.values[0]+before_df_t_1_max)/2

"""
statistic features
"""


def peak(r, data, col):
    if r['QueryOrder'] == 1:
        return
    else:
        before_df = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder']), col]
        return before_df.max()


def end(r, data, col):
    if r['QueryOrder'] == 1:
        return
    else:
        before_df = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] == r['QueryOrder'] - 1), col]
        return before_df.values[0]


def ave(r, data, col):
    if r['QueryOrder'] == 1:
        return
    else:
        before_df = data.loc[
            (data['SessionID'] == r['SessionID']) & (data['QueryOrder'] < r['QueryOrder']), col]
        return before_df.mean()


def model_simulation(data, model_name, target_col, base_cols, reference_cols, path, repeat=500):
    import warnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    auc_baselines = []
    auc_references = []
    auc_diffs = []

    acc_baselines = []
    acc_references = []
    acc_diffs = []

    recall_baselines = []
    recall_references = []
    recall_diffs = []

    f1_score_baselines = []
    f1_score_references = []
    f1_score_diffs = []
    for seed in tqdm(range(repeat)):
        X = data.loc[data['QueryOrder'] != 1, base_cols].values
        Y = data.loc[data['QueryOrder'] != 1, target_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_test_sparse = pd.get_dummies(y_test).values

        if model_name == 'lr':
            model = LogisticRegression()
        elif model_name == 'lgb':
            model = LGBMClassifier()
        elif model_name == 'rfc':
            model = RandomForestClassifier()
        elif model_name == 'xgb':
            model = XGBClassifier()
        elif model_name == 'knn':
            model = KNeighborsClassifier()
        elif model_name == 'nb':
            model = GaussianNB()
        elif model_name == 'dt':
            model = DecisionTreeClassifier()
        else:
            model = SVC(probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred_label = model.predict(X_test)
        # print(f'auc: {roc_auc_score(y_test_sparse, y_pred, average="weighted")}')
        # print(f'acc: {accuracy_score(y_test, y_pred_label)}')
        # print(f'recall: {recall_score(y_test, y_pred_label, average="weighted")}')
        # print(f'f1_score: {f1_score(y_test, y_pred_label, average="weighted")}')
        auc1 = roc_auc_score(y_test_sparse, y_pred, average="weighted")
        acc1 = accuracy_score(y_test, y_pred_label)
        recall1 = recall_score(y_test, y_pred_label, average="weighted")
        f1_score1 = f1_score(y_test, y_pred_label, average="weighted")

        X = data.loc[data['QueryOrder'] != 1, reference_cols].values
        Y = data.loc[data['QueryOrder'] != 1, 'target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_test_sparse = pd.get_dummies(y_test).values

        if model_name == 'lr':
            model = LogisticRegression()
        elif model_name == 'lgb':
            model = LGBMClassifier()
        elif model_name == 'rfc':
            model = RandomForestClassifier()
        elif model_name == 'xgb':
            model = XGBClassifier()
        elif model_name == 'knn':
            model = KNeighborsClassifier()
        elif model_name == 'nb':
            model = GaussianNB()
        elif model_name == 'dt':
            model = DecisionTreeClassifier()
        else:
            model = SVC(probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred_label = model.predict(X_test)
        auc2 = roc_auc_score(y_test_sparse, y_pred, average="weighted")
        acc2 = accuracy_score(y_test, y_pred_label)
        recall2 = recall_score(y_test, y_pred_label, average="weighted")
        f1_score2 = f1_score(y_test, y_pred_label, average="weighted")

        auc_baselines.append(auc1)
        auc_references.append(auc2)
        auc_diffs.append(auc2 - auc1)

        acc_baselines.append(acc1)
        acc_references.append(acc2)
        acc_diffs.append(acc2 - acc1)

        recall_baselines.append(recall1)
        recall_references.append(recall2)
        recall_diffs.append(recall2 - recall1)

        f1_score_baselines.append(f1_score1)
        f1_score_references.append(f1_score2)
        f1_score_diffs.append(f1_score2 - f1_score1)

    columns = ['auc_baseline', 'auc_reference', 'auc_diff', 'auc_win_rate',
               'acc_baseline', 'acc_reference', 'acc_diff', 'acc_win_rate',
               'recall_baseline', 'recall_reference', 'recall_diff', 'recall_win_rate',
               'f1_score_baseline', 'f1_score_reference', 'f1_score_diff', 'f1_score_win_rate']
    result = [str(round(np.array(auc_baselines).mean(), 6)) + '+/' + str(round(np.array(auc_baselines).std(), 6)),
              str(round(np.array(auc_references).mean(), 6)) + '+/' + str(round(np.array(auc_references).std(), 6)),
              str(round(np.array(auc_diffs).mean(), 6)) + '+/' + str(round(np.array(auc_diffs).std(), 6)),
              round((np.array(auc_diffs) > 0).mean(), 6),
              str(round(np.array(acc_baselines).mean(), 6)) + '+/' + str(round(np.array(acc_baselines).std(), 6)),
              str(round(np.array(acc_references).mean(), 6)) + '+/' + str(round(np.array(acc_references).std(), 6)),
              str(round(np.array(acc_diffs).mean(), 6)) + '+/' + str(round(np.array(acc_diffs).std(), 6)),
              round((np.array(acc_diffs) > 0).mean(), 6),
              str(round(np.array(recall_baselines).mean(), 6)) + '+/' + str(round(np.array(recall_baselines).std(), 6)),
              str(round(np.array(recall_references).mean(), 6)) + '+/' + str(
                  round(np.array(recall_references).std(), 6)),
              str(round(np.array(recall_diffs).mean(), 6)) + '+/' + str(round(np.array(recall_diffs).std(), 6)),
              round((np.array(recall_diffs) > 0).mean(), 6),
              str(round(np.array(f1_score_baselines).mean(), 6)) + '+/' + str(
                  round(np.array(f1_score_baselines).std(), 6)),
              str(round(np.array(f1_score_references).mean(), 6)) + '+/' + str(
                  round(np.array(f1_score_references).std(), 6)),
              str(round(np.array(f1_score_diffs).mean(), 6)) + '+/' + str(round(np.array(f1_score_diffs).std(), 6)),
              round((np.array(f1_score_diffs) > 0).mean(), 6)]
    index = [model_name]
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/scores'):
        os.makedirs(path + '/scores')
    if not os.path.exists(path + '/results'):
        os.makedirs(path + '/results')
    np.save(path + '/scores/' + model_name + 'auc_baselines.npy', np.array(auc_baselines))
    np.save(path + '/scores/' + model_name + 'acc_baselines.npy', np.array(acc_baselines))
    np.save(path + '/scores/' + model_name + 'recall_baselines.npy', np.array(recall_baselines))
    np.save(path + '/scores/' + model_name + 'f1_score_baselines.npy', np.array(f1_score_baselines))
    np.save(path + '/scores/' + model_name + 'auc_references.npy', np.array(auc_references))
    np.save(path + '/scores/' + model_name + 'acc_references.npy', np.array(acc_references))
    np.save(path + '/scores/' + model_name + 'recall_references.npy', np.array(recall_references))
    np.save(path + '/scores/' + model_name + 'f1_score_references.npy', np.array(f1_score_references))

    result_df = pd.DataFrame(np.array(result).reshape((1, -1)), index=index, columns=columns)
    result_df.to_csv(path + '/results/' + model_name + 'result.csv')
    return result_df