import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
from utils import model_simulation
from ast import literal_eval


reference_cols_t_1 = pd.read_csv('data/data_with_refer_feature/reference_cols_t_1.csv', index_col=['dataset'])
statistic_cols = pd.read_csv('data/data_with_refer_feature/statistic_cols.csv', index_col=['dataset'])

Session_2014 = pd.read_csv('data/data_with_refer_feature/Session_2014_refer_t_1.csv')
Session_2014_now_cols = ['QueryOrder', 'QueryLength']
Session_2014_past_peak_cols = literal_eval(statistic_cols.loc['Session2014', 'past_peak_cols'])
Session_2014_past_ave_cols = literal_eval(statistic_cols.loc['Session2014', 'past_ave_cols'])
Session_2014_past_end_cols = literal_eval(statistic_cols.loc['Session2014', 'past_end_cols'])
Session_2014_references1 = literal_eval(reference_cols_t_1.loc['Session2014', 'references1_t_1'])
Session_2014_references2 = literal_eval(reference_cols_t_1.loc['Session2014', 'references2_t_1'])
Session_2014_references3 = literal_eval(reference_cols_t_1.loc['Session2014', 'references3_t_1'])
Session_2014_references4 = literal_eval(reference_cols_t_1.loc['Session2014', 'references4_t_1'])
Session_2014_references5 = literal_eval(reference_cols_t_1.loc['Session2014', 'references5_t_1'])
Session_2014_references = [Session_2014_references1, Session_2014_references2, Session_2014_references3,
                           Session_2014_references4, Session_2014_references5]
def convert_to_class(r):
    if r<=-30:
        return 0
    elif r<=0:
        return 1
    elif r<=30:

        return 2
    else:
        return 3
Session_2014['target'] = Session_2014['delta_QueryDwellTime'].apply(convert_to_class)

results1 = pd.DataFrame()
results2 = pd.DataFrame()
results3 = pd.DataFrame()
results4 = pd.DataFrame()
results5 = pd.DataFrame()
results_df = [results1, results2, results3, results4, results5]

for col in Session_2014_now_cols+Session_2014_past_peak_cols+Session_2014_past_ave_cols+ \
           Session_2014_past_end_cols+Session_2014_references1+Session_2014_references2+ \
           Session_2014_references3+Session_2014_references4+Session_2014_references5:
    Session_2014.loc[np.isinf(Session_2014[col]), col] = 0
    upper_bound = np.percentile(Session_2014[col], 99)
    lower_bound = np.percentile(Session_2014[col], 1)
    Session_2014.loc[Session_2014[col] <= lower_bound, col] = lower_bound
    Session_2014.loc[Session_2014[col] >= upper_bound, col] = upper_bound
    Session_2014[col].fillna(0, inplace=True)

for i in range(5):
    print('reference'+str(i+1))
    for model_name in ['lr', 'lgb', 'rfc', 'xgb', 'knn', 'dt']:
        print(f'开始实验{model_name}')
        print('*'*50)
        result_df = model_simulation(Session_2014, model_name, 'target', Session_2014_now_cols + Session_2014_past_ave_cols,
                                     Session_2014_now_cols + Session_2014_past_ave_cols +
                                     Session_2014_past_peak_cols + Session_2014_past_end_cols +
                                     Session_2014_references[i], 'results/TREC-Session2014/delta_QueryDwellTime/reference'+str(i+1))
        results_df[i] = pd.concat([results_df[i], result_df], axis=0)
    results_df[i].to_csv('results/TREC-Session2014/delta_QueryDwellTime/results_reference'+str(i+1)+'.csv')

"""
=================================================================================================
Session_2013
=================================================================================================
"""
Session_2013 = pd.read_csv('data/data_with_refer_feature/Session_2013_refer_t_1.csv')
Session_2013_now_cols = ['QueryOrder', 'QueryLength']
Session_2013_past_peak_cols = literal_eval(statistic_cols.loc['Session2013', 'past_peak_cols'])
Session_2013_past_ave_cols = literal_eval(statistic_cols.loc['Session2013', 'past_ave_cols'])
Session_2013_past_end_cols = literal_eval(statistic_cols.loc['Session2013', 'past_end_cols'])
Session_2013_references1 = literal_eval(reference_cols_t_1.loc['Session2013', 'references1_t_1'])
Session_2013_references2 = literal_eval(reference_cols_t_1.loc['Session2013', 'references2_t_1'])
Session_2013_references3 = literal_eval(reference_cols_t_1.loc['Session2013', 'references3_t_1'])
Session_2013_references4 = literal_eval(reference_cols_t_1.loc['Session2013', 'references4_t_1'])
Session_2013_references5 = literal_eval(reference_cols_t_1.loc['Session2013', 'references5_t_1'])
Session_2013_references = [Session_2013_references1, Session_2013_references2, Session_2013_references3,
                           Session_2013_references4, Session_2013_references5]

def convert_to_class(r):
    if r <= 0:
        return 0
    else:
        return 1
Session_2013['target'] = Session_2013['delta_QueryDwellTime'].apply(convert_to_class)

results1 = pd.DataFrame()
results2 = pd.DataFrame()
results3 = pd.DataFrame()
results4 = pd.DataFrame()
results5 = pd.DataFrame()
results_df = [results1, results2, results3, results4, results5]

for col in Session_2013_now_cols+Session_2013_past_peak_cols+Session_2013_past_ave_cols+ \
           Session_2013_past_end_cols+Session_2013_references1+Session_2013_references2+ \
           Session_2013_references3+Session_2013_references4+Session_2013_references5:
    Session_2013.loc[np.isinf(Session_2013[col]), col] = 0
    upper_bound = np.percentile(Session_2013[col], 99)
    lower_bound = np.percentile(Session_2013[col], 1)
    Session_2013.loc[Session_2013[col] <= lower_bound, col] = lower_bound
    Session_2013.loc[Session_2013[col] >= upper_bound, col] = upper_bound
    Session_2013[col].fillna(0, inplace=True)

for i in range(5):
    print('reference'+str(i+1))
    for model_name in ['lr', 'lgb', 'rfc', 'xgb', 'knn', 'dt']:
        print(f'开始实验{model_name}')
        print('*'*50)
        result_df = model_simulation(Session_2013, model_name, 'target', Session_2013_now_cols + Session_2013_past_ave_cols,
                                     Session_2013_now_cols + Session_2013_past_ave_cols +
                                     Session_2013_past_peak_cols + Session_2013_past_end_cols +
                                     Session_2013_references[i], 'results/TREC-Session2013/delta_QueryDwellTime/reference'+str(i+1))
        results_df[i] = pd.concat([results_df[i], result_df], axis=0)
    results_df[i].to_csv('results/TREC-Session2013/delta_QueryDwellTime/results_reference'+str(i+1)+'.csv')
"""
================================================================================================
THU-KDD19
================================================================================================
"""
THU_KDD19 = pd.read_csv('data/data_with_refer_feature/THU_KDD19_refer_t_1.csv')
THU_KDD19_now_cols = ['QueryOrder', 'QueryLength']
THU_KDD19_past_peak_cols = literal_eval(statistic_cols.loc['KDD19', 'past_peak_cols'])
THU_KDD19_past_ave_cols = literal_eval(statistic_cols.loc['KDD19', 'past_ave_cols'])
THU_KDD19_past_end_cols = literal_eval(statistic_cols.loc['KDD19', 'past_end_cols'])
THU_KDD19_references1 = literal_eval(reference_cols_t_1.loc['KDD19', 'references1_t_1'])
THU_KDD19_references2 = literal_eval(reference_cols_t_1.loc['KDD19', 'references2_t_1'])
THU_KDD19_references3 = literal_eval(reference_cols_t_1.loc['KDD19', 'references3_t_1'])
THU_KDD19_references4 = literal_eval(reference_cols_t_1.loc['KDD19', 'references4_t_1'])
THU_KDD19_references5 = literal_eval(reference_cols_t_1.loc['KDD19', 'references5_t_1'])
THU_KDD19_references = [THU_KDD19_references1, THU_KDD19_references2, THU_KDD19_references3,
                        THU_KDD19_references4, THU_KDD19_references5]

def convert_to_class(r):
    if r <= 0:
        return 0
    else:
        return 1

THU_KDD19['target'] = THU_KDD19['delta_QueryDwellTime'].apply(convert_to_class)

results1 = pd.DataFrame()
results2 = pd.DataFrame()
results3 = pd.DataFrame()
results4 = pd.DataFrame()
results5 = pd.DataFrame()
results_df = [results1, results2, results3, results4, results5]

for col in THU_KDD19_now_cols+THU_KDD19_past_peak_cols+THU_KDD19_past_ave_cols+ \
           THU_KDD19_past_end_cols+THU_KDD19_references1+THU_KDD19_references2+ \
           THU_KDD19_references3+THU_KDD19_references4+THU_KDD19_references5:
    THU_KDD19.loc[np.isinf(THU_KDD19[col]), col] = 0
    upper_bound = np.percentile(THU_KDD19[col], 99)
    lower_bound = np.percentile(THU_KDD19[col], 1)
    THU_KDD19.loc[THU_KDD19[col] <= lower_bound, col] = lower_bound
    THU_KDD19.loc[THU_KDD19[col] >= upper_bound, col] = upper_bound
    THU_KDD19[col].fillna(0, inplace=True)

for i in range(5):
    print('reference'+str(i+1))
    for model_name in ['lr', 'lgb', 'rfc', 'xgb', 'knn', 'dt']:
        print(f'开始实验{model_name}')
        print('*'*50)
        result_df = model_simulation(THU_KDD19, model_name, 'target', THU_KDD19_now_cols + THU_KDD19_past_ave_cols,
                                     THU_KDD19_now_cols + THU_KDD19_past_ave_cols +
                                     THU_KDD19_past_peak_cols + THU_KDD19_past_end_cols +
                                     THU_KDD19_references[i], 'results/THU_KDD19/delta_QueryDwellTime/reference'+str(i+1))
        results_df[i] = pd.concat([results_df[i], result_df], axis=0)
    results_df[i].to_csv('results/THU_KDD19/delta_QueryDwellTime/results_reference'+str(i+1)+'.csv')