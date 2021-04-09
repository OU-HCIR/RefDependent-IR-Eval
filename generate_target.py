import pandas as pd

Session_2014 = pd.read_csv('data/TREC-Session2014/Session2014_total_feature.csv')
Session_2014_target_cols = ['QueryDwellTime', 'AvgRelScore', 'Clicks@5', 'ClickDepth']

for col in Session_2014_target_cols:
    Session_2014[col+'_shift1'] = Session_2014[col].shift(1)
    Session_2014['delta_'+col] = Session_2014[col+'_shift1'] - Session_2014[col]
    Session_2014.loc[Session_2014['QueryOrder'] == 1, 'delta_'+col] = 0
    Session_2014.drop([col+'_shift1'], axis=1, inplace=True)
Session_2014.to_csv('data/TREC-Session2014/Session2014_total_feature.csv')
"""
==========================================================================================
Session2013
==========================================================================================
"""
Session_2013 = pd.read_csv('data/TREC-Session2013/Session2013_total_feature.csv')
Session_2013_target_cols = ['QueryDwellTime', 'AvgRelScore', 'Clicks@5', 'ClickDepth']

for col in Session_2013_target_cols:
    Session_2013[col+'_shift1'] = Session_2013[col].shift(1)
    Session_2013['delta_'+col] = Session_2013[col+'_shift1'] - Session_2013[col]
    Session_2013.loc[Session_2013['QueryOrder'] == 1, 'delta_'+col] = 0
    Session_2013.drop([col+'_shift1'], axis=1, inplace=True)
Session_2013.to_csv('data/TREC-Session2013/Session2013_total_feature.csv')
"""
================================================================================================
THU-KDD19
================================================================================================
"""
THU_KDD19 = pd.read_csv('data/THU-KDD19/KDD19_total_feature.csv')
THU_KDD19_target_cols = ['QueryDwellTime', 'AveQueryRelScore', 'AveTaskRelScore', 'Clicks@5', 'ClickDepth']

for col in THU_KDD19_target_cols:
    THU_KDD19[col+'_shift1'] = THU_KDD19[col].shift(1)
    THU_KDD19['delta_'+col] = THU_KDD19[col+'_shift1'] - THU_KDD19[col]
    THU_KDD19.loc[THU_KDD19['QueryOrder'] == 1, 'delta_'+col] = 0
    THU_KDD19.drop([col+'_shift1'], axis=1, inplace=True)
THU_KDD19.to_csv('data/THU-KDD19/KDD19_total_feature.csv')