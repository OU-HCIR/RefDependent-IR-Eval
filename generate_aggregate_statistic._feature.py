import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import peak, ave, end

Session_2014_base_cols = ['QueryLength', 'QueryDwellTime', 'NewTerm', 'QuerySim',
                         'ClickCount', 'KeyDocCount', 'RelDocCount', 'AvgContent',
                         'TotalContent', 'AveClickRank', 'ClickDepth', 'SERPtime',
                        'RR', 'Clicks@3', 'Clicks@5', 'Clicks@5+', 'nDCG@3', 'nDCG@5',
                        'nDCG@10', 'Precision@3', 'Precision@5', 'Precision@10',
                        'Cost-Benefit-1', 'Cost-Benefit-1_1', 'Cost-Benefit-1_2',
                        'Cost-Benefit-2', 'Cost-Benefit-3']
Session_2014 = pd.read_csv('data/TREC-Session2014/Session2014_total_feature.csv')
Session_2014_past_peak_cols = []
Session_2014_past_end_cols = []
Session_2014_past_ave_cols = []

for col in tqdm(Session_2014_base_cols):
    Session_2014[col+'_peak'] = Session_2014.progress_apply(peak, args=(Session_2014, col,), axis=1)
    Session_2014[col+'_ave'] = Session_2014.progress_apply(ave, args=(Session_2014, col,), axis=1)
    Session_2014[col+'_end'] = Session_2014.progress_apply(end, args=(Session_2014, col,), axis=1)

    Session_2014_past_peak_cols.append(col+'_peak')
    Session_2014_past_ave_cols.append(col+'_ave')
    Session_2014_past_end_cols.append(col+'_end')
Session_2014.to_csv('data/TREC-Session2014/Session2014_total_feature.csv')
"""
=================================================================================================
Session_2013
=================================================================================================
"""
Session_2013_base_cols = ['QueryLength', 'QueryDwellTime', 'NewTerm', 'QuerySim',
                         'ClickCount', 'KeyDocCount', 'RelDocCount', 'AvgContent',
                         'TotalContent', 'AveClickRank', 'ClickDepth', 'SERPtime',
                        'RR', 'Clicks@3', 'Clicks@5', 'Clicks@5+', 'nDCG@3', 'nDCG@5',
                        'nDCG@10', 'Precision@3', 'Precision@5', 'Precision@10',
                          'AvgRelScore']
Session_2013 = pd.read_csv('data/TREC-Session2013/Session2013_total_feature.csv')

Session_2013_past_peak_cols = []
Session_2013_past_end_cols = []
Session_2013_past_ave_cols = []

for col in tqdm(Session_2013_base_cols):
    Session_2013[col+'_peak'] = Session_2013.progress_apply(peak, args=(Session_2013, col,), axis=1)
    Session_2013[col+'_ave'] = Session_2013.progress_apply(ave, args=(Session_2013, col,), axis=1)
    Session_2013[col+'_end'] = Session_2013.progress_apply(end, args=(Session_2013, col,), axis=1)

    Session_2013_past_peak_cols.append(col+'_peak')
    Session_2013_past_ave_cols.append(col+'_ave')
    Session_2013_past_end_cols.append(col+'_end')
Session_2013.to_csv('data/TREC-Session2013/Session2013_total_feature.csv')
"""
================================================================================================
THU-KDD19
================================================================================================
"""
THU_KDD19_base_cols = ['QueryLength', 'QueryDwellTime', 'NewTerm', 'QuerySim',
                       'ClickCount', 'QueryRelDocCount', 'QueryKeyDocCount',
                       'TaskRelDocCount', 'TaskKeyDocCount', 'QueryNDCG@3', 'QueryNDCG@5',
                       'QueryNDCG@10', 'TaskNDCG@3', 'TaskNDCG@5', 'TaskNDCG@10',
                       'QueryPrecision@3', 'QueryPrecision@5', 'QueryPrecision@10',
                       'TaskPrecision@3', 'TaskPrecision@5', 'TaskPrecision@10',
                       'Clicks@3', 'Clicks@5', 'Clicks@10', 'AveQueryRelScore',
                       'AveTaskRelScore', 'NewTerm', 'QuerySim', 'ClickDepth']
THU_KDD19 = pd.read_csv('data/THU-KDD19/KDD19_total_feature.csv')

THU_KDD19_past_peak_cols = []
THU_KDD19_past_end_cols = []
THU_KDD19_past_ave_cols = []

for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col+'_peak'] = THU_KDD19.progress_apply(peak, args=(THU_KDD19, col,), axis=1)
    THU_KDD19[col+'_ave'] = THU_KDD19.progress_apply(ave, args=(THU_KDD19, col,), axis=1)
    THU_KDD19[col+'_end'] = THU_KDD19.progress_apply(end, args=(THU_KDD19, col,), axis=1)

    THU_KDD19_past_peak_cols.append(col+'_peak')
    THU_KDD19_past_ave_cols.append(col+'_ave')
    THU_KDD19_past_end_cols.append(col+'_end')
THU_KDD19.to_csv('data/THU-KDD19/KDD19_total_feature.csv')

statistic_cols = {'dataset': ['Session2014', 'Session2013', 'KDD19'],
                  'past_peak_cols': [Session_2014_past_peak_cols, Session_2013_past_peak_cols, THU_KDD19_past_peak_cols],
                  'past_ave_cols': [Session_2014_past_ave_cols, Session_2013_past_ave_cols, THU_KDD19_past_ave_cols],
                  'past_end_cols': [Session_2014_past_end_cols, Session_2013_past_end_cols, THU_KDD19_past_end_cols]}
statistic_cols = pd.DataFrame(statistic_cols)
statistic_cols.to_csv('data/data_with_refer_feature/statistic_cols.csv')