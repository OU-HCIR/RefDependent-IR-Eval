"""
The feature value at time t-1 minus various statistical values of the feature at time 0 to t-2.
We use sum, peak, end, ave as the representation of people's psychological expectation
"""
import pandas as pd
from tqdm import tqdm
from utils import first_reference_t_1, peak_reference_t_1, ave_reference_t_1, end_reference_t_1, peak_end_reference_t_1
print("generate ref t-1 feature for Session2014")

Session_2014_base_cols = ['QueryLength', 'NewTerm','QuerySim', 'NumClickedPage', 'ClickCount',
                'AveClickRank', 'Clicks@3', 'Clicks@5', 'Clicks@5+', 'ClickDepth', 'NumKeyDoc',
                'NumRelevantDoc', 'SERPtime', 'AvgContent', 'TotalContent', 'QueryDwellTime',
                'TimetoFirstLastClick', 'ClickPrecision' , 'ratiofirstclick', 'RR',
                'DCG3', 'DCG5','DCG10', 'nDCG@3', 'nDCG@5', 'nDCG@10', 'RelDocCount', 'KeyDocCount',
                'ReformulationTime', 'Precision@3', 'Precision@5', 'Precision@10', 'AvgRelScore',
               'Cost-Benefit-1', 'Cost-Benefit-1_1', 'Cost-Benefit-1_2', 'Cost-Benefit-2', 'Cost-Benefit-3']
Session_2014 = pd.read_csv('data/TREC-Session2014/Session2014_total_feature.csv')

Session_2014_references1 = []
for col in tqdm(Session_2014_base_cols):
    Session_2014[col+'_reference1'] = Session_2014.apply(first_reference_t_1, args=(Session_2014, col,), axis=1)
    Session_2014_references1.append(col+'_reference1')

Session_2014_references2 = []
for col in tqdm(Session_2014_base_cols):
    Session_2014[col + '_reference2'] = Session_2014.apply(peak_reference_t_1, args=(Session_2014, col,), axis=1)
    Session_2014_references2.append(col + '_reference2')

Session_2014_references3 = []
for col in tqdm(Session_2014_base_cols):
    Session_2014[col+'_reference3'] = Session_2014.apply(ave_reference_t_1, args=(Session_2014, col,), axis=1)
    Session_2014_references3.append(col+'_reference3')

Session_2014_references4 = []
for col in tqdm(Session_2014_base_cols):
    Session_2014[col+'_reference4'] = Session_2014.apply(end_reference_t_1, args=(Session_2014, col,), axis=1)
    Session_2014_references4.append(col+'_reference4')

Session_2014_references5 = []
for col in tqdm(Session_2014_base_cols):
    Session_2014[col+'_reference5'] = Session_2014.apply(peak_end_reference_t_1, args=(Session_2014, col,), axis=1)
    Session_2014_references5.append(col+'_reference5')

for col in Session_2014_references1+Session_2014_references2+Session_2014_references3+Session_2014_references4+Session_2014_references5:
    Session_2014[col].fillna(0, inplace=True)

Session_2014.to_csv('data/data_with_refer_feature/Session_2014_refer_t_1.csv', index=False)
"""
=================================================================================================
Session_2013
=================================================================================================
"""
print("generate ref t-1 feature for Session2013")

Session_2013_base_cols = ['QueryLength', 'QueryDwellTime', 'NewTerm', 'QuerySim',
                         'ClickCount', 'KeyDocCount', 'RelDocCount', 'AvgContent',
                         'TotalContent', 'AveClickRank', 'ClickDepth', 'SERPtime',
                        'RR', 'Clicks@3', 'Clicks@5', 'Clicks@5+', 'DCG@3', 'DCG@5',
                        'DCG@10', 'nDCG@3', 'nDCG@5', 'nDCG@10', 'Precision@3',
                        'Precision@5', 'Precision@10', 'AvgRelScore']
Session_2013 = pd.read_csv('data/TREC-Session2013/Session2013_total_feature.csv')

Session_2013_references1 = []
for col in tqdm(Session_2013_base_cols):
    Session_2013[col+'_reference1'] = Session_2013.apply(first_reference_t_1, args=(Session_2013, col,), axis=1)
    Session_2013_references1.append(col+'_reference1')

Session_2013_references2 = []
for col in tqdm(Session_2013_base_cols):
    Session_2013[col + '_reference2'] = Session_2013.apply(peak_reference_t_1, args=(Session_2013, col,), axis=1)
    Session_2013_references2.append(col + '_reference2')

Session_2013_references3 = []
for col in tqdm(Session_2013_base_cols):
    Session_2013[col+'_reference3'] = Session_2013.apply(ave_reference_t_1, args=(Session_2013, col,), axis=1)
    Session_2013_references3.append(col+'_reference3')

Session_2013_references4 = []
for col in tqdm(Session_2013_base_cols):
    Session_2013[col+'_reference4'] = Session_2013.apply(end_reference_t_1, args=(Session_2013, col,), axis=1)
    Session_2013_references4.append(col+'_reference4')

Session_2013_references5 = []
for col in tqdm(Session_2013_base_cols):
    Session_2013[col+'_reference5'] = Session_2013.apply(peak_end_reference_t_1, args=(Session_2013, col,), axis=1)
    Session_2013_references5.append(col+'_reference5')

for col in Session_2013_references1+Session_2013_references2+Session_2013_references3+Session_2013_references4+Session_2013_references5:
    Session_2013[col].fillna(0, inplace=True)

Session_2013.to_csv('data/data_with_refer_feature/Session_2013_refer_t_1.csv', index=False)
"""
================================================================================================
THU-KDD19
================================================================================================
"""

print("generate ref t-1 feature for THU-KDD19")
THU_KDD19_base_cols = ['QueryLength', 'QueryDwellTime', 'BEGIN_SEARCH_count', 'JUMP_IN_count',
                       'MOUSE_MOVE_count', 'JUMP_OUT_count', 'QUERY_REFORM_count',
                       'HoverCount', 'CLICK_count', 'ScrollDist', 'OVER_count', 'SEARCH_END_count',
                       'GOTO_PAGE_count', 'ClickCount', 'QueryRelDocCount', 'QueryKeyDocCount',
                       'TaskRelDocCount', 'TaskKeyDocCount', 'QueryNDCG@3', 'QueryNDCG@5',
                       'QueryNDCG@10', 'TaskNDCG@3', 'TaskNDCG@5', 'TaskNDCG@10',
                       'QueryDCG@3', 'QueryDCG@5', 'QueryDCG@10', 'TaskDCG@3', 'TaskDCG@5', 'TaskDCG@10',
                       'QueryPrecision@3', 'QueryPrecision@5', 'QueryPrecision@10',
                       'TaskPrecision@3', 'TaskPrecision@5', 'TaskPrecision@10', 'Clicks@3',
                       'Clicks@5', 'Clicks@10', 'NewTerm', 'QuerySim',
                       'AveQueryRelScore', 'AveTaskRelScore', 'click_precision_query',
                       'click_precision_task', 'ClickDepth', 'ActionCount',
                       'TimeFirstClick', 'TimeLastClick', 'TotalContent', 'AvgContent',
                       'SERPtime', 'AvgClickRank', 'Query_Cost-Benefit-1',
                       'Task_Cost-Benefit-1', 'Query_Cost-Benefit-2', 'Task_Cost-Benefit-2',
                       'Query_Cost-Benefit-3', 'Task_Cost-Benefit-3']
THU_KDD19 = pd.read_csv('data/THU-KDD19/KDD19_total_feature.csv')

THU_KDD19_references1 = []
for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col+'_reference1'] = THU_KDD19.apply(first_reference_t_1, args=(THU_KDD19, col,), axis=1)
    THU_KDD19_references1.append(col+'_reference1')

THU_KDD19_references2 = []
for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col + '_reference2'] = THU_KDD19.apply(peak_reference_t_1, args=(THU_KDD19, col,), axis=1)
    THU_KDD19_references2.append(col + '_reference2')

THU_KDD19_references3 = []
for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col+'_reference3'] = THU_KDD19.apply(ave_reference_t_1, args=(THU_KDD19, col,), axis=1)
    THU_KDD19_references3.append(col+'_reference3')

THU_KDD19_references4 = []
for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col+'_reference4'] = THU_KDD19.apply(end_reference_t_1, args=(THU_KDD19, col,), axis=1)
    THU_KDD19_references4.append(col+'_reference4')

THU_KDD19_references5 = []
for col in tqdm(THU_KDD19_base_cols):
    THU_KDD19[col+'_reference5'] = THU_KDD19.apply(peak_end_reference_t_1, args=(THU_KDD19, col,), axis=1)
    THU_KDD19_references5.append(col+'_reference5')

for col in THU_KDD19_references1+THU_KDD19_references2+THU_KDD19_references3+THU_KDD19_references4+THU_KDD19_references5:
    THU_KDD19[col].fillna(0, inplace=True)

THU_KDD19.to_csv('data/data_with_refer_feature/THU_KDD19_refer_t_1.csv')

reference_cols_t_1 = {'dataset': ['Session2014', 'Session2013', 'KDD19'],
                      'references1_t_1': [Session_2014_references1, Session_2013_references1, THU_KDD19_references1],
                      'references2_t_1': [Session_2014_references2, Session_2013_references2, THU_KDD19_references2],
                      'references3_t_1': [Session_2014_references3, Session_2013_references3, THU_KDD19_references3],
                      'references4_t_1': [Session_2014_references4, Session_2013_references4, THU_KDD19_references4],
                      'references5_t_1': [Session_2014_references5, Session_2013_references5, THU_KDD19_references5]}
reference_cols_t_1 = pd.DataFrame(reference_cols_t_1)
reference_cols_t_1.to_csv('data/data_with_refer_feature/reference_cols_t_1.csv', index=False)