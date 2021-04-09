import pandas as pd
from tqdm import tqdm, tqdm_notebook
import warnings
from utils import calc_dcg, calc_ndcg
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
tqdm.pandas()

# generate feature for session 2014
print("generate feature for session 2014")
raw_data = pd.read_excel('data/TREC-Session2014/all_sessiondata_2014_raw_data_modify20200102.xlsx')
querySegment = pd.read_csv('data/TREC-Session2014/querySegment_20200101_SegmentID_SessionBegin_SessionEnd.csv')
relevant_judgement = pd.read_table('data/TREC-Session2014/relevance judgment.txt', sep=' ', header=None, names=['topic_num', 'col2', 'clueweb12id', 'relevant_score'])
# preprocess data
raw_data['Attribute:userid'] = raw_data['Attribute:userid'].fillna(-1)
raw_data['interaction.results.result.title'] = raw_data['interaction.results.result.title'].fillna(' ')
raw_data['interaction.results.result.snippet'] = raw_data['interaction.results.result.snippet'].fillna(' ')
raw_data['Query'] = raw_data['interaction.query'].values

querySegment['AveDwellTime'] = querySegment['AveDwellTime'].fillna(0)
querySegment['AveClickRank'] = querySegment['AveClickRank'].fillna(0)
querySegment['ClickDepth'] = querySegment['ClickDepth'].fillna(0)
querySegment['SearchEngineDwellTime'] = querySegment['SearchEngineDwellTime'].fillna(0)
querySegment['ClickPrecision'] = querySegment['ClickPrecision'].fillna(0)
querySegment.loc[querySegment['TimetoFirstLastClick'].isnull(), 'TimetoFirstLastClick'] = querySegment.loc[querySegment['TimetoFirstLastClick'].isnull(), 'QueryDwellTime']
querySegment.loc[querySegment['ratiofirstclick']=='#DIV/0!', 'ratiofirstclick'] = 0
querySegment['ratiofirstclick'] = querySegment['ratiofirstclick'].astype(float)

numerical_columns = ['QueryOrder', 'QueryLen', 'QueryDwellTime', 'NumClickedPage',
                     'NumClickedDoc', 'NumKeyDoc', 'NumRelevantDoc', 'NumKeyDoc_real', 'NumRelevantDoc_real','AveDwellTime',
                     'TotalDwellTime', 'AveClickRank', 'ClickDepth', 'SearchEngineDwellTime',
                     'ClickPrecision', 'TimetoFirstLastClick', 'ReciproalRank', 'ratiofirstclick', 'ReformulationTime']
classical_columns = ['TopicProduct', 'TopicGoal', 'TaskType']

for col in classical_columns:
    map_dict = dict(zip(querySegment[col].unique(), range(len(querySegment[col].unique()))))
    querySegment[col] = querySegment[col].map(map_dict)

# get relevant score and convert it to binary variables
relevant_judgement['topic_num_clueweb12id'] = relevant_judgement['topic_num'].astype(str) + '_' + relevant_judgement['clueweb12id']
raw_data['interaction.clicked.click.docno_topic_num_clueweb12id'] = raw_data['topic.Attribute:num'].astype(str) + '_' + raw_data['interaction.clicked.click.docno']
map_dict = relevant_judgement.loc[:,['topic_num_clueweb12id', 'relevant_score']]
map_dict.set_index(['topic_num_clueweb12id'], inplace=True)
raw_data['interaction.clicked.click.docno_relevent_score'] = raw_data['interaction.clicked.click.docno_topic_num_clueweb12id'].map(map_dict.to_dict()['relevant_score'])
raw_data['interaction.clicked.click.docno_relevent_binary'] = raw_data['interaction.clicked.click.docno_relevent_score'].apply(lambda r: 0 if r<=0 else 1)
raw_data['interaction.clicked.click.docno_relevent_key'] = raw_data['interaction.clicked.click.docno_relevent_score'].apply(lambda r: 0 if r<=1 else 1)
raw_data['is_clicked'] = 0
raw_data.loc[raw_data['interaction.clicked.click.docno'].isnull()==False, "is_clicked"] = 1
raw_data['interaction.clicked.click.docno_relevent_binary_clicked'] = raw_data['interaction.clicked.click.docno_relevent_binary'] * raw_data['is_clicked']
raw_data['interaction.clicked.click.docno_relevent_key_clicked'] = raw_data['interaction.clicked.click.docno_relevent_key'] * raw_data['is_clicked']
raw_data['interaction.clicked.click.docno_relevent_binary_clicked'].fillna(0, inplace=True)
raw_data['interaction.clicked.click.docno_relevent_key_clicked'].fillna(0, inplace=True)

# generate RelDocCount and KeyDocCount
def NumDoc(r, col1, col2):
    return r.loc[r[col1]==1, col2].nunique()
my_feature = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: NumDoc(r, 'interaction.clicked.click.docno_relevent_binary_clicked', 'interaction.clicked.click.docno'))
my_feature = my_feature.reset_index(drop=False)
my_feature.rename(columns={0:'NumRelevantDoc'}, inplace=True)
num_key_doc = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: NumDoc(r, 'interaction.clicked.click.docno_relevent_key_clicked', 'interaction.clicked.click.docno'))
my_feature['NumKeyDoc'] = num_key_doc.values

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
my_feature['reformulate_starttime'] = segment_start_time.values
my_feature.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
my_feature.reset_index(drop=True, inplace=True)
querySegment['RelDocCount'] = my_feature['NumRelevantDoc']
querySegment['KeyDocCount'] = my_feature['NumKeyDoc']

# DCG@i, i=3 5 10
raw_data['interaction.results.result.topic_num_clueweb12id'] = raw_data['topic.Attribute:num'].astype(str) + '_' + raw_data['interaction.results.result.clueweb12id']
map_dict = relevant_judgement.loc[:,['topic_num_clueweb12id', 'relevant_score']]
map_dict.set_index(['topic_num_clueweb12id'], inplace=True)
raw_data['interaction.results.result.relevent_score'] = raw_data['interaction.results.result.topic_num_clueweb12id'].map(map_dict.to_dict()['relevant_score'])
raw_data['interaction.results.result.relevent_binary'] = raw_data['interaction.results.result.relevent_score'].apply(lambda r: 0 if r<=0 else 1)

DCG3 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_dcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 3))
DCG3 = DCG3.reset_index(drop=False)
DCG3.rename(columns={0:'DCG3'}, inplace=True)

DCG5 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_dcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 5))
DCG5 = DCG5.reset_index(drop=False)
DCG5.rename(columns={0:'DCG5'}, inplace=True)

DCG10 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_dcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 10))
DCG10 = DCG10.reset_index(drop=False)
DCG10.rename(columns={0:'DCG10'}, inplace=True)

DCG_feature = pd.concat([DCG3, DCG5.loc[:,['DCG5']], DCG10.loc[:,['DCG10']]], axis=1)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
DCG_feature['reformulate_starttime'] = segment_start_time.values
DCG_feature.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
DCG_feature.reset_index(drop=True, inplace=True)

querySegment['DCG3'] = DCG_feature['DCG3']
querySegment['DCG5'] = DCG_feature['DCG5']
querySegment['DCG10'] = DCG_feature['DCG10']

DCG_cols = ['DCG3', 'DCG5', 'DCG10']

querySegment['DCG3'].fillna(0, inplace=True)
querySegment['DCG5'].fillna(0, inplace=True)
querySegment['DCG10'].fillna(0, inplace=True)

# nDCG@i, i=3 5 10
NDCG3 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_ndcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 3))
NDCG3 = NDCG3.reset_index(drop=False)
NDCG3.rename(columns={0:'NDCG3'}, inplace=True)

NDCG5 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_ndcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 5))
NDCG5 = NDCG5.reset_index(drop=False)
NDCG5.rename(columns={0:'NDCG5'}, inplace=True)

NDCG10 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: calc_ndcg(r, 'interaction.results.result.Attribute:rank', 'interaction.results.result.relevent_score', 10))
NDCG10 = NDCG10.reset_index(drop=False)
NDCG10.rename(columns={0:'NDCG10'}, inplace=True)

NDCG_feature = pd.concat([NDCG3, NDCG5.loc[:,['NDCG5']], NDCG10.loc[:,['NDCG10']]], axis=1)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
NDCG_feature['reformulate_starttime'] = segment_start_time.values
NDCG_feature.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
NDCG_feature.reset_index(drop=True, inplace=True)

querySegment['nDCG@3'] = NDCG_feature['NDCG3']
querySegment['nDCG@5'] = NDCG_feature['NDCG5']
querySegment['nDCG@10'] = NDCG_feature['NDCG10']

NDCG_cols = ['nDCG@3', 'nDCG@5', 'nDCG@10']

querySegment['nDCG@3'].fillna(0, inplace=True)
querySegment['nDCG@5'].fillna(0, inplace=True)
querySegment['nDCG@10'].fillna(0, inplace=True)

# Precision@i, i=3 5 10
def Precision(r, relevant_col, i):
    return len(r.loc[r[relevant_col]==1, :])/len(r)

Precision3 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: Precision(r, 'interaction.results.result.relevent_binary', 3))
Precision3 = Precision3.reset_index(drop=False)
Precision3.rename(columns={0:'Precision3'}, inplace=True)

Precision5 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: Precision(r, 'interaction.results.result.relevent_binary', 5))
Precision5 = Precision5.reset_index(drop=False)
Precision5.rename(columns={0:'Precision5'}, inplace=True)

Precision10 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: Precision(r, 'interaction.results.result.relevent_binary', 10))
Precision10 = Precision10.reset_index(drop=False)
Precision10.rename(columns={0:'Precision10'}, inplace=True)

Precision_feature = pd.concat([Precision3, Precision5.loc[:,['Precision5']], Precision10.loc[:,['Precision10']]], axis=1)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
Precision_feature['reformulate_starttime'] = segment_start_time.values
Precision_feature.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
Precision_feature.reset_index(drop=True, inplace=True)

querySegment['Precision@3'] = Precision_feature['Precision3']
querySegment['Precision@5'] = Precision_feature['Precision5']
querySegment['Precision@10'] = Precision_feature['Precision10']

Precision_cols = ['Precision@3', 'Precision@5', 'Precision@10']

querySegment['Precision@3'].fillna(0, inplace=True)
querySegment['Precision@5'].fillna(0, inplace=True)
querySegment['Precision@10'].fillna(0, inplace=True)

# reciprocal rank
def Reciprocal_Rank(r, clicked_relevant_col):
    if len(r.loc[r[clicked_relevant_col]==1])>0:
        return r.loc[r[clicked_relevant_col]==1, 'interaction.results.result.Attribute:rank'].values[0]
    else:
        return 11

ReciprocalRank = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: Reciprocal_Rank(r, 'interaction.results.result.relevent_binary'))
ReciprocalRank = ReciprocalRank.reset_index(drop=False)
ReciprocalRank.rename(columns={0:'ReciprocalRank'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
ReciprocalRank['reformulate_starttime'] = segment_start_time.values
ReciprocalRank.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
ReciprocalRank.reset_index(drop=True, inplace=True)

querySegment['RR'] = ReciprocalRank['ReciprocalRank']

Precision_cols.append('RR')

querySegment['RR'].fillna(0, inplace=True)

"""
Cost Benefit Measure 1: if no click, value = 0; if click != 0, value = ∑▒〖[REL*content dwell time]〗/query dwell time [nominator represents benefit, denominator represents total cost]
In these measures: C-B 1 & C-B 1-1, if REL content dwell time >30s, then assume that the time = 30s, which means additional time on the same content page will NOT generate more benefits)
In addition to 30s, perhaps some other thresholds can be tested as well, e.g. 45s, 60s, depending on data distribution)
"""
def CBM1(r):
    r = r.drop_duplicates(['interaction.clicked.click.Attribute:starttime'])
    r['contentTime'] = r['interaction.clicked.click.Attribute:endtime']-r['interaction.clicked.click.Attribute:starttime']
    r['CB1ContentWeight'] = r.apply(lambda r: 30 if (r['contentTime']>30 and r['interaction.clicked.click.docno_relevent_score']<0) else r['contentTime'], axis=1)
    r['CBF1'] = (r['CB1ContentWeight'] * r['interaction.clicked.click.docno_relevent_score'])/(r['interaction.Attribute:endtime']-r['interaction.Attribute:starttime'])
    return r['CBF1'].sum()

CBF1 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(CBM1)
CBF1 = CBF1.reset_index(drop=False)
CBF1.rename(columns={0:'CBF1'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
CBF1['reformulate_starttime'] = segment_start_time.values
CBF1.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
CBF1.reset_index(drop=True, inplace=True)
querySegment['Cost-Benefit-1'] = CBF1['CBF1']
"""
========================================================================================================
C-B 1-1: ∑▒〖[REL*content dwell time]〗 
"""
def CBM1_1(r):
    r = r.drop_duplicates(['interaction.clicked.click.Attribute:starttime'])
    r['contentTime'] = r['interaction.clicked.click.Attribute:endtime']-r['interaction.clicked.click.Attribute:starttime']
    r['CB1ContentWeight'] = r.apply(lambda r: 30 if (r['contentTime']>30 and r['interaction.clicked.click.docno_relevent_score']<0) else r['contentTime'], axis=1)
    r['CBF1_1'] = (r['CB1ContentWeight'] * r['interaction.clicked.click.docno_relevent_score'])
    return r['CBF1_1'].sum()

CBF1_1 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(CBM1_1)
CBF1_1 = CBF1_1.reset_index(drop=False)
CBF1_1.rename(columns={0:'CBF1_1'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
CBF1_1['reformulate_starttime'] = segment_start_time.values
CBF1_1.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
CBF1_1.reset_index(drop=True, inplace=True)
querySegment['Cost-Benefit-1_1'] = CBF1_1['CBF1_1']
"""
=========================================================================================================

"""
def CBM1_2(r):
    r = r.drop_duplicates(['interaction.clicked.click.Attribute:starttime'])
    r['contentTime'] = r['interaction.clicked.click.Attribute:endtime']-r['interaction.clicked.click.Attribute:starttime']
    r['CB1ContentWeight'] = r.apply(lambda r: 30 if (r['contentTime']>30 and r['interaction.clicked.click.docno_relevent_score']<0) else r['contentTime'], axis=1)
    r['CBF1_2'] = (r.loc[r['interaction.clicked.click.docno_relevent_score']<0, 'CB1ContentWeight'] * r.loc[r['interaction.clicked.click.docno_relevent_score']<0, 'interaction.clicked.click.docno_relevent_score'])
    return -r['CBF1_2'].sum()

CBF1_2 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(CBM1_2)
CBF1_2 = CBF1_2.reset_index(drop=False)
CBF1_2.rename(columns={0:'CBF1_2'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
CBF1_2['reformulate_starttime'] = segment_start_time.values
CBF1_2.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
CBF1_2.reset_index(drop=True, inplace=True)
querySegment['Cost-Benefit-1_2'] = CBF1_2['CBF1_2']

"""
计算最大content time，方便进行search failure的判断
"""
def MaxContentTime(r):
    r = r.drop_duplicates(['interaction.clicked.click.Attribute:starttime'])
    r['contentTime'] = r['interaction.clicked.click.Attribute:endtime'] - r[
        'interaction.clicked.click.Attribute:starttime']
    return r['contentTime'].max()
max_content_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(MaxContentTime)
max_content_time = max_content_time.reset_index(drop=False)
max_content_time.rename(columns={0: 'MaxContent'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
max_content_time['reformulate_starttime'] = segment_start_time.values
max_content_time.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
max_content_time.reset_index(drop=True, inplace=True)
querySegment['MaxContent'] = max_content_time['MaxContent']
querySegment['MaxContent'].fillna(0, inplace=True)
"""
=======================================================================================
Cost Benefit Measure 2: click precision = number of relevant pages clicked/total number of pages clicked;
Cost Benefit Measure 3: number of relevant pages clicked/click depth
Cost Benefit Measure 4: (number of relevant pages clicked/click depth) * SERP dwell time
"""
querySegment['Cost-Benefit-2'] = querySegment['RelDocCount']/(querySegment['ClickDepth']+1e-12)
querySegment['Cost-Benefit-3'] = querySegment['Cost-Benefit-2'] * querySegment['SearchEngineDwellTime']

CB_cols = ['Cost-Benefit-1', 'Cost-Benefit-1_1', 'Cost-Benefit-1_2', 'Cost-Benefit-2', 'Cost-Benefit-3']

"""
exit condition
"""
averageRelevanceScore = raw_data.groupby(['SegmentID'])['interaction.clicked.click.docno_relevent_score'].apply(lambda r: r[r.isnull()==False].mean())
querySegment['AvgRelScore'] = querySegment['SegmentID'].map(averageRelevanceScore)
querySegment['AvgRelScore'].fillna(0, inplace=True)

raw_data['interaction.clicked.click.Attribute:dwelltime'] = raw_data['interaction.clicked.click.Attribute:endtime'] - raw_data['interaction.clicked.click.Attribute:starttime']
raw_data['interaction.clicked.click.Attribute:dwelltime_30'] = raw_data['interaction.clicked.click.Attribute:dwelltime'].apply(lambda r: 1 if r>=30 else 0)
raw_data['irrelevant_docs_30+s'] = raw_data['interaction.clicked.click.Attribute:dwelltime_30'] * (1-raw_data['interaction.clicked.click.docno_relevent_binary_clicked'])
irrelevantDocs30Num = raw_data.groupby(['SegmentID'])['irrelevant_docs_30+s'].sum()
querySegment['irrelevantDocs30Num'] = querySegment['SegmentID'].map(irrelevantDocs30Num)
querySegment['irrelevantDocs30Num'].fillna(0, inplace=True)

exit_condition_cols = ['AvgRelScore', 'irrelevantDocs30Num']

"""
target: new_unique_query_terms (new unique terms/total number of unique terms)
"""
def new_unique_query_terms_num(r):
    if r['isSegmentStart']==1:
        return 0
    now_query_list = r['Query'].split()
    last_query_list = r['Query_shift1'].split()
    total_unique_terms = set(now_query_list+last_query_list)
    new_unique_terms = set([term for term in now_query_list if (term not in last_query_list)])
    return len(new_unique_terms)/len(total_unique_terms)

querySegment['Query_shift1'] = querySegment['Query'].shift(1)
querySegment['new_unique_query_terms_num'] = querySegment.progress_apply(new_unique_query_terms_num, axis=1)

"""
target: query similarity (shared unique terms/total number of unique terms)
"""
def query_similarity(r):
    if r['isSegmentStart']==1:
        return 0
    now_query_list = r['Query'].split()
    last_query_list = r['Query_shift1'].split()
    total_unique_terms = set(now_query_list+last_query_list)
    shared_unique_terms = set([term for term in now_query_list if (term in last_query_list)])
    return len(shared_unique_terms)/len(total_unique_terms)

querySegment['Query_shift1'] = querySegment['Query'].shift(1)
querySegment['QuerySim'] = querySegment.progress_apply(query_similarity, axis=1)

"""
target: num of clicks rank 1-i
"""
def NumOfClicks(r, start=0, end=3):
    clicked_doc = r['interaction.clicked.click.docno'].tolist()
    candidate_doc = set(r.loc[(r['interaction.results.result.Attribute:rank']>=start)&(r['interaction.results.result.Attribute:rank']<=end), 'interaction.results.result.clueweb12id'].tolist())
    count = 0
    for doc in candidate_doc:
        if doc in clicked_doc:
            count+=1
    return count
NumOfClick3 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: NumOfClicks(r, 0, 3))
NumOfClick3 = NumOfClick3.reset_index(drop=False)
NumOfClick3.rename(columns={0:'NumOfClick3'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
NumOfClick3['reformulate_starttime'] = segment_start_time.values
NumOfClick3.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
NumOfClick3.reset_index(drop=True, inplace=True)
querySegment['Clicks@3'] = NumOfClick3['NumOfClick3']

NumOfClick5 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: NumOfClicks(r, 0, 5))
NumOfClick5 = NumOfClick5.reset_index(drop=False)
NumOfClick5.rename(columns={0:'NumOfClick5'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
NumOfClick5['reformulate_starttime'] = segment_start_time.values
NumOfClick5.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
NumOfClick5.reset_index(drop=True, inplace=True)
querySegment['Clicks@5'] = NumOfClick5['NumOfClick5']

NumOfClick6 = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num']).apply(lambda r: NumOfClicks(r, 6, 10))
NumOfClick6 = NumOfClick6.reset_index(drop=False)
NumOfClick6.rename(columns={0:'NumOfClick6'}, inplace=True)

segment_start_time = raw_data.groupby(['Query', 'Attribute:userid', 'Attribute:num'])['interaction.Attribute:starttime'].apply(lambda r: r.values[0])
NumOfClick6['reformulate_starttime'] = segment_start_time.values
NumOfClick6.sort_values(['Attribute:num', 'reformulate_starttime'], inplace=True)
NumOfClick6.reset_index(drop=True, inplace=True)
querySegment['Clicks@5+'] = NumOfClick6['NumOfClick6']

querySegment = querySegment.rename(columns={'QueryLen': 'QueryLength',
                                            'NumClickedDoc': 'ClickCount',
                                            'AveDwellTime': 'AvgContent',
                                            'TotalDwellTime': 'TotalContent',
                                            'SearchEngineDwellTime': 'SERPtime',
                                            'isSegmentEnd': 'isSessionEnd',
                                            'new_unique_query_terms_num': 'NewTerm'})
querySegment.to_csv('data/TREC-Session2014/Session2014_total_feature.csv', index=False)

"""
============================================================================================================
Session 2013
============================================================================================================
"""
print("generate feature for session 2013")
raw_data = pd.read_excel('data/TREC-Session2013/sessiontrack2013_data.xlsx')
relevanceJudgment = pd.read_table('data/TREC-Session2013/relevanceJudgment03.txt',header=None, names=['/session/topic/@num', 'col2', 'clueweb12id', 'score'], sep=' ')

def click_clueweb(r):
    condition = (raw_data['/session/interaction/query']==r['/session/interaction/query'])& \
                (raw_data['/session/interaction/#id']==r['/session/interaction/#id'])& \
                (raw_data['/session/interaction/@num']==r['/session/interaction/@num'])& \
                (raw_data['/session/interaction/results/result/@rank']==r['/session/interaction/clicked/click/rank'])
    try:
        return raw_data.loc[condition, '/session/interaction/results/result/clueweb12id'].values[0]
    except:
        return

raw_data['/session/interaction/clicked/click/clueweb12id'] = raw_data.progress_apply(click_clueweb, axis=1)

raw_data['/session/topic/@num_/session/interaction/clicked/click/clueweb12id'] = raw_data['/session/topic/@num'].astype(str) + '_' + raw_data['/session/interaction/clicked/click/clueweb12id']
relevanceJudgment['/session/topic/@num_clueweb12id'] = relevanceJudgment['/session/topic/@num'].astype(str) + '_' + relevanceJudgment['clueweb12id']
map_dict = relevanceJudgment.loc[:, ['/session/topic/@num_clueweb12id', 'score']]
map_dict.set_index('/session/topic/@num_clueweb12id', inplace=True)
raw_data['/session/interaction/clicked/click/score'] = raw_data['/session/topic/@num_/session/interaction/clicked/click/clueweb12id'].map(map_dict.to_dict()['score'])

raw_data['/session/interaction/clicked/click/relevant_doc'] = raw_data['/session/interaction/clicked/click/score'].apply(lambda r: 1 if r>=1 else 0)
raw_data['/session/interaction/clicked/click/key_doc'] = raw_data['/session/interaction/clicked/click/score'].apply(lambda r: 1 if r>=2 else 0)

raw_data.sort_values(['/session/@num', '/session/interaction/@starttime'], inplace=True)
min_map_dict = raw_data.groupby(['/session/@num'])['/session/interaction/@num'].min()
raw_data['query_order'] = raw_data['/session/interaction/@num'] - raw_data['/session/@num'].map(min_map_dict) + 1
raw_data['clicked_dwell_time'] = raw_data['/session/interaction/clicked/click/@endtime'] - raw_data['/session/interaction/clicked/click/@starttime']
raw_data['clicked_dwell_time'] = raw_data['/session/interaction/clicked/click/@endtime'] - raw_data['/session/interaction/clicked/click/@starttime']
raw_data['clicked_dwell_time'].fillna(0, inplace=True)
raw_data['MaxContent'] = raw_data['clicked_dwell_time']  # 为了计算search failure进行准备
raw_data['clicked_rank'] = raw_data['/session/interaction/clicked/click/rank']
raw_data['num_of_click3'] = raw_data['/session/interaction/clicked/click/rank']
raw_data['num_of_click5'] = raw_data['/session/interaction/clicked/click/rank']
raw_data['num_of_click10'] = raw_data['/session/interaction/clicked/click/rank']

raw_data['clicked_rank'] = raw_data['/session/interaction/clicked/click/rank']

# generate main feature
def ave_click_rank(r):
    if len(r)==0:
        return
    else:
        return r[r.isnull()==False].mean()
def max_click_rank(r):
    if len(r)==0:
        return
    else:
        return r[r.isnull()==False].max()
def num_of_click(r, start=1, end=3):
    top_i = r[(r>=start)&(r<=end)].nunique()
    return top_i

query_segment = raw_data.groupby(['/session/interaction/#id']).agg({'/session/@num': lambda r: r.values[0],
                                                    '/session/topic/@num': lambda r: r.values[0],
                                                    '/session/interaction/query': lambda r: r.values[0],
                                                    'query_order': lambda r: r.values[0],
                                                    '/session/interaction/@starttime': lambda r: r.values[0],
                                                    '/session/currentquery/@starttime': lambda r: r.values[0],
                                                    '/session/interaction/clicked/click/clueweb12id': lambda r: len(set(r[r.isnull()==False].values.tolist())),
                                                    '/session/interaction/clicked/click/relevant_doc': lambda r: r.values.sum(),
                                                    '/session/interaction/clicked/click/key_doc': lambda r: r.values.sum(),
                                                    'clicked_dwell_time': lambda r: r.values.sum(),
                                                    'MaxContent': lambda r: r.values.max(),
                                                    '/session/interaction/clicked/click/rank':ave_click_rank,
                                                    'clicked_rank': max_click_rank,
                                                    'num_of_click3': lambda r: num_of_click(r, 1, 3),
                                                    'num_of_click5': lambda r: num_of_click(r, 1, 5),
                                                    'num_of_click10': lambda r: num_of_click(r,6, 10)
                                                    })
query_segment.reset_index(drop=False, inplace=True)
query_segment.rename(columns={
    '/session/interaction/#id':'segment_id',
    '/session/@num':'session_id',
    '/session/topic/@num':'topic_id',
    '/session/interaction/query':'query',
    '/session/interaction/@starttime':'segment_starttime',
    '/session/currentquery/@starttime':'session_end_time',
    '/session/interaction/clicked/click/clueweb12id':'num_click_doc',
    '/session/interaction/clicked/click/relevant_doc':'num_relevant_doc',
    '/session/interaction/clicked/click/key_doc':'num_key_doc',
    'clicked_dwell_time':'total_dwell_time',
    '/session/interaction/clicked/click/rank':'ave_click_rank',
    'clicked_rank':'click_depth'
}, inplace=True)

query_segment['segment_endtime'] = query_segment['segment_starttime'].shift(-1)
query_segment['session_max_order'] = query_segment['session_id'].map(query_segment.groupby(['session_id'])['query_order'].max())
query_segment['session_end'] = 0
query_segment.loc[query_segment['query_order']==query_segment['session_max_order'], 'session_end'] = 1
query_segment.loc[query_segment['query_order']==query_segment['session_max_order'], 'segment_endtime'] = query_segment.loc[query_segment['query_order']==query_segment['session_max_order'], 'session_end_time']
query_segment['query_dwell_time'] = query_segment['segment_endtime'] - query_segment['segment_starttime']
query_segment['search_engine_dwell_time'] = query_segment['query_dwell_time'] - query_segment['total_dwell_time']
query_segment['ave_dwell_time']  = 0
query_segment.loc[query_segment['num_click_doc']!=0, 'ave_dwell_time'] = query_segment.loc[query_segment['num_click_doc']!=0, 'total_dwell_time']/query_segment.loc[query_segment['num_click_doc']!=0, 'num_click_doc']
query_segment['click_precision']  = query_segment.loc[query_segment['num_click_doc']!=0, 'num_relevant_doc']/query_segment.loc[query_segment['num_click_doc']!=0, 'num_click_doc']
query_segment['time_to_first_last_click'] = query_segment['total_dwell_time']
query_segment.loc[query_segment['time_to_first_last_click'].isnull(), 'time_to_first_last_click'] = query_segment.loc[query_segment['time_to_first_last_click'].isnull(), 'query_dwell_time']
query_segment.drop(['session_max_order'], axis=1, inplace=True)
query_segment['query_len'] = query_segment['query'].apply(lambda r: len(r))

def ReciproalRank(r):
    condition = (raw_data['/session/interaction/#id']==r['segment_id'])& \
                (raw_data['/session/interaction/clicked/click/relevant_doc']==1)
    try:
        return raw_data[condition, '/session/interaction/clicked/click/rank'].min()
    except:
        return 11
def time_first_click(r):
    condition = (raw_data['/session/interaction/#id']==r['segment_id'])& \
                (raw_data['/session/interaction/clicked/click/rank'].isnull()==False)
    if len(raw_data.loc[condition])==0:
        return
    else:
        return raw_data.loc[condition, '/session/interaction/clicked/click/@starttime'].min()
query_segment['reciporal_rank'] = query_segment.progress_apply(ReciproalRank, axis=1)
query_segment['time_first_click'] = query_segment.progress_apply(time_first_click, axis=1)
query_segment['ratio_first_click'] = query_segment['time_first_click']/query_segment['query_dwell_time']
query_segment['ratio_first_click'].fillna(1, inplace=True)


query_segment['ave_click_rank'].fillna(11, inplace=True)
query_segment['click_depth'].fillna(0, inplace=True)
query_segment['search_engine_dwell_time'].fillna(0, inplace=True)
query_segment['click_precision'].fillna(0, inplace=True)
query_segment['query_dwell_time'].fillna(0, inplace=True)

average_relevance_score = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/clicked/click/score'].apply(lambda r: r[r.isnull()==False].mean())
query_segment['average_relevance_score'] = query_segment['segment_id'].map(average_relevance_score)
query_segment['average_relevance_score'].fillna(0, inplace=True)

raw_data['click_irevelant_30+s'] = raw_data['clicked_dwell_time'].apply(lambda r: 1 if r>=30 else 0) * (1-raw_data['/session/interaction/clicked/click/relevant_doc'])
irrelevant_docs_30num = raw_data.groupby(['/session/interaction/#id']).apply(lambda r: r.loc[r['click_irevelant_30+s']==1, '/session/topic/@num_/session/interaction/clicked/click/clueweb12id'].nunique())
query_segment['irrelevant_docs_30num'] = query_segment['segment_id'].map(irrelevant_docs_30num)
query_segment['irrelevant_docs_30num'].fillna(0, inplace=True)

def new_unique_query_terms_num(query, last_query):
    last_query = last_query.split()
    query = query.split()
    new_unique_term = [word for word in query if (word not in last_query)]
    total_unique_term = [word for word in query] + [word for word in last_query]
    return len(set(new_unique_term))/len(set(total_unique_term))
query_segment['last_query'] = query_segment['query'].shift(1)
query_segment.loc[query_segment['query_order']==1 , 'last_query'] = ''
query_segment['new_unique_query_terms_num'] = query_segment.apply(lambda r: new_unique_query_terms_num(r['query'], r['last_query']), axis=1)

def last_query_similarity(query, last_query):
    last_query = last_query.split()
    query = query.split()
    shared_unique_term = [word for word in query if (word in last_query)]
    total_unique_term = [word for word in query] + [word for word in last_query]
    return len(set(shared_unique_term))/len(set(total_unique_term))
query_segment['last_query'] = query_segment['query'].shift(1)
query_segment.loc[query_segment['query_order']==1 , 'last_query'] = ''
query_segment['query_similarity'] = query_segment.apply(lambda r: last_query_similarity(r['query'], r['last_query']), axis=1)

relevant_score = pd.read_table('data/TREC-Session2013/relevanceJudgment03.txt', sep=' ' ,header=None, names=['topic_id', 'col1', 'clueweb12id', 'relevant_score'])
raw_data['topic_id_clueweb12id'] = raw_data['/session/topic/@num'].astype(str) + '_' + raw_data['/session/interaction/results/result/clueweb12id']
relevant_score['topic_id_clueweb12id'] = relevant_score['topic_id'].astype(str) + '_' + relevant_score['clueweb12id']
map_dict = relevant_score.loc[:, ['topic_id_clueweb12id', 'relevant_score']]
map_dict.set_index(['topic_id_clueweb12id'], inplace=True)
raw_data['relevant_score'] = raw_data['topic_id_clueweb12id'].map(map_dict.to_dict()['relevant_score'])

dcg3 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_dcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 3))
query_segment['dcg3'] = query_segment['segment_id'].map(dcg3)
dcg5 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_dcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 5))
query_segment['dcg5'] = query_segment['segment_id'].map(dcg5)
dcg10 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_dcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 10))
query_segment['dcg10'] = query_segment['segment_id'].map(dcg10)

ndcg3 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_ndcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 3))
query_segment['ndcg3'] = query_segment['segment_id'].map(ndcg3)
ndcg5 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_ndcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 5))
query_segment['ndcg5'] = query_segment['segment_id'].map(ndcg5)
ndcg10 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: calc_ndcg(r, '/session/interaction/results/result/@rank', 'relevant_score', 10))
query_segment['ndcg10'] = query_segment['segment_id'].map(ndcg10)

"""precision"""
def precision(r, relevant_score, i):
    try:
        return len(r.loc[(r[relevant_score]>=1)&(r['/session/interaction/results/result/@rank']<=i)])/len(r.loc[r['/session/interaction/results/result/@rank']<=i])
    except:
        return

precision3 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: precision(r, 'relevant_score', 3))
query_segment['precision3'] = query_segment['segment_id'].map(precision3)
precision5 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: precision(r, 'relevant_score', 5))
query_segment['precision5'] = query_segment['segment_id'].map(precision5)
precision10 = raw_data.groupby(['/session/interaction/#id'])['/session/interaction/results/result/@rank', 'relevant_score'].apply(lambda r: precision(r, 'relevant_score', 10))
query_segment['precision10'] = query_segment['segment_id'].map(precision10)

query_segment = query_segment.rename(columns={'query_order': 'QueryOrder',
                                              'num_click_doc': 'ClickCount',
                                              'num_relevant_doc': 'RelDocCount',
                                              'num_key_doc': 'KeyDocCount',
                                              'total_dwell_time': 'TotalContent',
                                              'ave_click_rank': 'AveClickRank',
                                              'click_depth': 'ClickDepth',
                                              'num_of_click3': 'Clicks@3',
                                              'num_of_click5': 'Clicks@5',
                                              'num_of_click10': 'Clicks@5+',
                                              'session_end': 'isSessionEnd',
                                              'query_dwell_time': 'QueryDwellTime',
                                              'search_engine_dwell_time': 'SERPtime',
                                              'ave_dwell_time': 'AvgContent',
                                              'query_len': 'QueryLength',
                                              'reciporal_rank': 'RR',
                                              'average_relevance_score': 'AvgRelScore',
                                              'new_unique_query_terms_num': 'NewTerm',
                                              'query_similarity': 'QuerySim',
                                              'dcg3': 'DCG@3',
                                              'dcg5': 'DCG@5',
                                              'dcg10': 'DCG@10',
                                              'ndcg3': 'nDCG@3',
                                              'ndcg5': 'nDCG@5',
                                              'ndcg10': 'nDCG@10',
                                              'precision3': 'Precision@3',
                                              'precision5': 'Precision@5',
                                              'precision10': 'Precision@10',
                                              'session_id': 'SessionID'})

query_segment.to_csv('data/TREC-Session2013/Session2013_total_feature.csv', index=False)

"""
=====================================================================================================
THU-KDD19
=====================================================================================================
"""
print("generate feature for THU-KDD19")

all_relevance_annotation = pd.read_table('data/THU-KDD19/all_relevance_annotation.txt')
anno_annotation = pd.read_csv('data/THU-KDD19/anno_annotation.csv')
anno_log = pd.read_csv('data/THU-KDD19/anno_log.csv')
anno_querysatisfaction = pd.read_csv('data/THU-KDD19/anno_querysatisfaction.csv')
anno_questionnaireanswer = pd.read_csv('data/THU-KDD19/anno_questionnaireanswer.csv')
anno_searchresult = pd.read_csv('data/THU-KDD19/anno_searchresult.csv')
anno_sessionannotation = pd.read_csv('data/THU-KDD19/anno_sessionannotation.csv')
anno_task = pd.read_csv('data/THU-KDD19/anno_task.csv')

anno_querysatisfaction.drop(['Unnamed: 0'], axis=1, inplace=True)
anno_sessionannotation.drop(['Unnamed: 0'], axis=1, inplace=True)

def extract_timestamp(r):
    r_list = r.split('\t')
    for string in r_list:
        if 'TIME' in string:
            try:
                time = int(string.split('=')[1])
                return time
            except:
                return
    return
anno_annotation['time_stamp'] = anno_annotation['content'].apply(extract_timestamp)
anno_log['time_stamp'] = anno_log['content'].apply(extract_timestamp)
anno_querysatisfaction['time_stamp'] = anno_querysatisfaction['content'].apply(extract_timestamp)
anno_questionnaireanswer['time_stamp'] = anno_questionnaireanswer['content'].apply(extract_timestamp)
anno_sessionannotation['time_stamp'] = anno_sessionannotation['content'].apply(extract_timestamp)

data = anno_querysatisfaction

def target(r):
    if r>=4:
        return 1
    else:
        return 0
data['target'] = data['score'].apply(target)
data['target'].value_counts()

data.sort_values(['studentID', 'task_id', 'time_stamp'], inplace=True)

# session_id
data['session_id'] = data.apply(
    lambda r: anno_sessionannotation.loc[(anno_sessionannotation['studentID'] == r['studentID']) &
                                         (anno_sessionannotation['task_id'] == r['task_id']), 'id'].values[0], axis=1)

# query_order
data['query_order'] = range(len(data))
session_min_order = data.groupby(['session_id'])['query_order'].min()
data['session_min_order'] = data['session_id'].map(session_min_order)
data['query_order'] -= data['session_min_order'] - 1
data.drop(['session_min_order'], axis=1, inplace=True)

# query len
data['query_len'] = data['query'].apply(lambda r: len(r))

# session_end
data['dummy_i'] = data['session_id'] * 1000 + data['query_order']
data['dummy_i_shiftdown'] = data['dummy_i'].shift(-1)
data['session_end'] = data.apply(lambda r: 1 if (r['dummy_i_shiftdown'] - r['dummy_i']) != 1 else 0, axis=1)
data.drop(['dummy_i', 'dummy_i_shiftdown'], axis=1, inplace=True)


# query_dwell_time
def query_dwell_time(r):
    condition = (anno_log['studentID'] == r['studentID']) & (anno_log['task_id'] == r['task_id']) & (
                anno_log['query'] == r['query'])
    try:
        dwell_time = anno_log.loc[condition & (anno_log['action'] == 'QUERY_REFORM'), 'time_stamp'].values[0] - \
                     anno_log.loc[condition & (anno_log['action'] == 'BEGIN_SEARCH'), 'time_stamp'].values[0]
        return dwell_time
    except:
        return anno_log.loc[condition, 'time_stamp'].max() - anno_log.loc[condition, 'time_stamp'].min()


data['query_dwell_time'] = data.progress_apply(query_dwell_time, axis=1)
data.loc[data['query_dwell_time'].isnull(), 'query_dwell_time'] = data.loc[data[
                                                                               'query_dwell_time'].isnull(), 'time_stamp'].shift(
    -1) - data.loc[data['query_dwell_time'].isnull(), 'time_stamp']
data.loc[data['query_dwell_time'].isnull(), 'query_dwell_time'] = data.loc[
    (data['studentID'] == 2014012759) & (data['task_id'] == 7), 'query_dwell_time'].mean()
data['query_dwell_time'] = data['query_dwell_time']/1000

# {action}_numbers
def action_count(r, action):
    condition = (anno_log['studentID'] == r['studentID']) & (anno_log['task_id'] == r['task_id']) & (
                anno_log['query'] == r['query'])
    return len(anno_log.loc[condition & (anno_log['action'] == action), :])


actions = ['BEGIN_SEARCH', 'JUMP_IN', 'MOUSE_MOVE', 'JUMP_OUT', 'QUERY_REFORM',
           'HOVER', 'CLICK', 'SCROLL', 'OVER', 'SEARCH_END', 'GOTO_PAGE']
action_cols = []
for action in tqdm(actions):
    data[action + '_count'] = data.apply(action_count, args=(action,), axis=1)
    action_cols.append(action + '_count')
data['ActionCount'] = data.loc[:, action_cols].sum(axis=1)
base_features = ['query_order', 'query_len', 'query_dwell_time']

"""
TimeFirstClick, TimeLastClick
"""
def segment_start_time(r):
    condition = (anno_log['studentID'] == r['studentID']) & (anno_log['task_id'] == r['task_id']) & (
                anno_log['query'] == r['query'])
    try:
        start_time = anno_log.loc[condition & (anno_log['action'] == 'BEGIN_SEARCH'), 'time_stamp'].values[0]
        return start_time
    except:
        return anno_log.loc[condition, 'time_stamp'].min()
data['segment_start_time'] = data.progress_apply(segment_start_time, axis=1)

session_start_time = data.groupby(['session_id'])['segment_start_time'].min()
data['session_start_time'] = data['session_id'].map(session_start_time)

def segment_first_click_time(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])&(anno_annotation['query']==r['query'])
    return anno_annotation.loc[condition,'time_stamp'].min()

def segment_last_click_time(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])&(anno_annotation['query']==r['query'])
    return anno_annotation.loc[condition,'time_stamp'].max()

data['segment_first_click_time'] = data.progress_apply(segment_first_click_time, axis=1)
data['segment_last_click_time'] = data.progress_apply(segment_last_click_time, axis=1)

session_first_click_time = data.groupby(['session_id'])['segment_first_click_time'].min()
session_last_click_time = data.groupby(['session_id'])['segment_last_click_time'].max()

data['session_first_click_time'] = data['session_id'].map(session_first_click_time)
data['session_last_click_time'] = data['session_id'].map(session_last_click_time)

data['TimeFirstClick'] = (data['session_first_click_time'] - data['session_start_time'])/1000
data['TimeLastClick'] = (data['session_last_click_time'] - data['session_start_time'])/1000

"""
TotalContent, AvgContent, SERPtime, ClickCount
"""
def clicked_doc_count(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])&(anno_annotation['query']==r['query'])
    return len(anno_annotation.loc[condition,:])
data['clicked_doc_count'] = data.apply(clicked_doc_count, axis=1)

def calculate_content_time(r, method='total'):
    condition = (anno_log['studentID'] == r['studentID']) & (anno_log['task_id'] == r['task_id']) & (
            anno_log['query'] == r['query'])
    jump_out_timestamp_list = anno_log.loc[condition & (anno_log['action'] == 'JUMP_OUT'), 'time_stamp'].tolist()
    jump_in_timestamp_list = anno_log.loc[condition & (anno_log['action'] == 'JUMP_IN'), 'time_stamp'].tolist()

    try:
        if (len(jump_out_timestamp_list) == len(jump_in_timestamp_list)) and (len(jump_out_timestamp_list) > 0):
            return (sum(jump_in_timestamp_list) - sum(jump_out_timestamp_list)) / 1000
        elif (len(jump_out_timestamp_list) == 0) or (len(jump_in_timestamp_list) == 0):
            return 0
        else:
            index_in = 0
            index_out = 0
            output = 0
            """
            左右依次向后移动，直到一侧到底为止，若in对应序列时间戳在out之后，则两者相减,
            同时索引加一，若in对应时间戳在out之前，则in继续加1，out不动
            """
            while index_in < len(jump_in_timestamp_list) and index_out < len(jump_out_timestamp_list):
                if jump_in_timestamp_list[index_in] >= jump_out_timestamp_list[index_out]:
                    if method == 'total':
                        output += jump_in_timestamp_list[index_in] - jump_out_timestamp_list[index_out]
                    elif method == 'max':
                        output = max(output, jump_in_timestamp_list[index_in] - jump_out_timestamp_list[index_out])
                    else:
                        raise ValueError('no such method!')
                    index_in += 1
                    index_out += 1
                else:
                    index_in += 1
            return output / 1000

    except:
        return 0


data['TotalContent'] = data.progress_apply(calculate_content_time, args=('total', ), axis=1)
data['MaxContent'] = data.progress_apply(calculate_content_time, args=('max', ), axis=1)
data['AvgContent'] = data.progress_apply(lambda r: r['TotalContent']/r['clicked_doc_count'] if r['clicked_doc_count']!=0 else 0, axis=1)
data['SERPtime'] = data.progress_apply(lambda r: r['query_dwell_time'] - r['TotalContent'] if (r['query_dwell_time']>r['TotalContent']) else 0, axis=1)


"""
relevance score
"""
def anno_annotation_score(r, score_type='query_relevance'):
    condition = (all_relevance_annotation['url']==r['result_url'])&(all_relevance_annotation['task']==r['task_id'])
    try:
        return all_relevance_annotation.loc[condition, score_type].values[0]
    except:
        return

anno_annotation['query_relevance_score'] = anno_annotation.progress_apply(anno_annotation_score, args=('query_relevance',), axis=1)
anno_annotation['task_relevance_score'] = anno_annotation.progress_apply(anno_annotation_score, args=('task_relevance',), axis=1)

anno_annotation['query_relevance'] = anno_annotation['query_relevance_score'].apply(lambda r: 1 if r>=1 else 0)
anno_annotation['query_key'] = anno_annotation['query_relevance_score'].apply(lambda r: 1 if r>=2 else 0)
anno_annotation['task_relevance'] = anno_annotation['task_relevance_score'].apply(lambda r: 1 if r>=1 else 0)
anno_annotation['task_key'] = anno_annotation['task_relevance_score'].apply(lambda r: 1 if r>=2 else 0)

"""
relevance
"""
def query_relevance_count(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])
    condition &= (anno_annotation['query']==r['query'])&(anno_annotation['query_relevance']==1)
    return len(anno_annotation.loc[condition,:])

def query_key_count(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])
    condition &= (anno_annotation['query']==r['query'])&(anno_annotation['query_key']==1)
    return len(anno_annotation.loc[condition,:])

def task_relevance_count(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])
    condition &= (anno_annotation['query']==r['query'])&(anno_annotation['task_relevance']==1)
    return len(anno_annotation.loc[condition,:])

def task_key_count(r):
    condition = (anno_annotation['studentID']==r['studentID'])&(anno_annotation['task_id']==r['task_id'])
    condition &= (anno_annotation['query']==r['query'])&(anno_annotation['task_key']==1)
    return len(anno_annotation.loc[condition,:])

data['query_relevance_count'] = data.apply(query_relevance_count, axis=1)
data['query_key_count'] = data.apply(query_key_count, axis=1)
data['task_relevance_count'] = data.apply(task_relevance_count, axis=1)
data['task_key_count'] = data.apply(task_key_count, axis=1)

relevance_cols = ['clicked_doc_count', 'query_relevance_count', 'query_key_count', 'task_relevance_count', 'task_key_count']

anno_searchresult['query_rank'] = anno_searchresult['query'] + '_' + anno_searchresult['rank'].astype(str)
all_relevance_annotation['query_rank'] = all_relevance_annotation['query'] + '_' + all_relevance_annotation['rank'].astype(str)
anno_searchresult = anno_searchresult.merge(all_relevance_annotation.loc[:,['query_rank', 'query_relevance', 'task_relevance']], how='left', on='query_rank')
use_search_result = anno_searchresult.loc[(anno_searchresult['query'].isin(data['query'].tolist()))&(anno_searchresult['rank']<10)]

query_nDCG3 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'query_relevance', 3))
query_nDCG3.rename(columns={0: 'query_nDCG3'}, inplace=True)
query_nDCG5 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'query_relevance', 5))
query_nDCG5.rename(columns={0: 'query_nDCG5'}, inplace=True)
query_nDCG10 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'query_relevance', 10))
query_nDCG10.rename(columns={0: 'query_nDCG10'}, inplace=True)

task_nDCG3 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'task_relevance', 3))
task_nDCG3.rename(columns={0: 'task_nDCG3'}, inplace=True)
task_nDCG5 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'task_relevance', 5))
task_nDCG5.rename(columns={0: 'task_nDCG5'}, inplace=True)
task_nDCG10 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_ndcg(r, 'rank', 'task_relevance', 10))
task_nDCG10.rename(columns={0: 'task_nDCG10'}, inplace=True)

data['query_nDCG3'] = data['query'].map(query_nDCG3)
data['query_nDCG5'] = data['query'].map(query_nDCG5)
data['query_nDCG10'] = data['query'].map(query_nDCG10)

data['task_nDCG3'] = data['query'].map(task_nDCG3)
data['task_nDCG5'] = data['query'].map(task_nDCG5)
data['task_nDCG10'] = data['query'].map(task_nDCG10)

data['query_nDCG3'].fillna(0, inplace=True)
data['query_nDCG5'].fillna(0, inplace=True)
data['query_nDCG10'].fillna(0, inplace=True)

data['task_nDCG3'].fillna(0, inplace=True)
data['task_nDCG5'].fillna(0, inplace=True)
data['task_nDCG10'].fillna(0, inplace=True)

query_DCG3 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'query_relevance', 3))
query_DCG3.rename(columns={0: 'query_DCG3'}, inplace=True)
query_DCG5 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'query_relevance', 5))
query_DCG5.rename(columns={0: 'query_DCG5'}, inplace=True)
query_DCG10 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'query_relevance', 10))
query_DCG10.rename(columns={0: 'query_DCG10'}, inplace=True)

task_DCG3 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'task_relevance', 3))
task_DCG3.rename(columns={0: 'task_DCG3'}, inplace=True)
task_DCG5 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'task_relevance', 5))
task_DCG5.rename(columns={0: 'task_DCG5'}, inplace=True)
task_DCG10 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(
    lambda r: calc_dcg(r, 'rank', 'task_relevance', 10))
task_DCG10.rename(columns={0: 'task_DCG10'}, inplace=True)

data['query_DCG3'] = data['query'].map(query_DCG3)
data['query_DCG5'] = data['query'].map(query_DCG5)
data['query_DCG10'] = data['query'].map(query_DCG10)

data['task_DCG3'] = data['query'].map(task_DCG3)
data['task_DCG5'] = data['query'].map(task_DCG5)
data['task_DCG10'] = data['query'].map(task_DCG10)

data['query_DCG3'].fillna(0, inplace=True)
data['query_DCG5'].fillna(0, inplace=True)
data['query_DCG10'].fillna(0, inplace=True)

data['task_DCG3'].fillna(0, inplace=True)
data['task_DCG5'].fillna(0, inplace=True)
data['task_DCG10'].fillna(0, inplace=True)

"""
precision
"""
def precision(r, relevant_score, i):
    return len(r.loc[(r[relevant_score]>=1)&(r['rank']<i)])/len(r.loc[r['rank']<i])

query_precision3 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(lambda r: precision(r, 'query_relevance', 3))
query_precision3.rename(columns={0:'query_precision3'},inplace=True)
query_precision5 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(lambda r: precision(r, 'query_relevance', 5))
query_precision5.rename(columns={0:'query_precision5'},inplace=True)
query_precision10 = use_search_result.groupby(['query'])['rank', 'query_relevance'].apply(lambda r: precision(r, 'query_relevance', 10))
query_precision10.rename(columns={0:'query_precision10'},inplace=True)

task_precision3 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(lambda r: precision(r, 'task_relevance', 3))
task_precision3.rename(columns={0:'task_precision3'},inplace=True)
task_precision5 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(lambda r: precision(r, 'task_relevance', 5))
task_precision5.rename(columns={0:'task_precision5'},inplace=True)
task_precision10 = use_search_result.groupby(['query'])['rank', 'task_relevance'].apply(lambda r: precision(r, 'task_relevance', 10))
task_precision10.rename(columns={0:'task_precision10'},inplace=True)

data['query_precision3'] = data['query'].map(query_precision3)
data['query_precision5'] = data['query'].map(query_precision5)
data['query_precision10'] = data['query'].map(query_precision10)

data['task_precision3'] = data['query'].map(task_precision3)
data['task_precision5'] = data['query'].map(task_precision5)
data['task_precision10'] = data['query'].map(task_precision10)

data['query_precision3'].fillna(0, inplace=True)
data['query_precision5'].fillna(0, inplace=True)
data['query_precision10'].fillna(0, inplace=True)

data['task_precision3'].fillna(0, inplace=True)
data['task_precision5'].fillna(0, inplace=True)
data['task_precision10'].fillna(0, inplace=True)
precision_cols = ['query_precision3', 'query_precision5', 'query_precision10', 'task_precision3', 'task_precision5', 'task_precision10']

"""
number of click@ 3, 5, 10
"""


def NumOfClick(r, i):
    click_data = anno_annotation.loc[(anno_annotation['studentID'] == r['studentID']) &
                                     (anno_annotation['task_id'] == r['task_id']) &
                                     (anno_annotation['query'] == r['query']) &
                                     (anno_annotation['rank'] < i)]
    click_data.drop_duplicates(['result_url'], inplace=True)
    return len(click_data)


anno_annotation['rank'] = anno_annotation['result_id'].progress_apply(lambda r: int(r.split('_')[-1]))
data['num_of_click3'] = data.progress_apply(NumOfClick, axis=1, args=(3,))
data['num_of_click5'] = data.progress_apply(NumOfClick, axis=1, args=(5,))
data['num_of_click10'] = data.progress_apply(NumOfClick, axis=1, args=(10,))

"""
AvgClickRank
"""
def avg_click_rank(r):
    click_data = anno_annotation.loc[(anno_annotation['studentID'] == r['studentID']) &
                                     (anno_annotation['task_id'] == r['task_id']) &
                                     (anno_annotation['query'] == r['query']) &
                                     (anno_annotation['rank'] < 10), :]
    click_data.drop_duplicates(['result_url'], inplace=True)
    try:
        return click_data['rank'].mean()
    except:
        return 0
data['AvgClickRank'] = data.progress_apply(avg_click_rank, axis=1)

"""
NewTerm
"""
def new_unique_query_terms_num(query, last_query):
    new_unique_term = [word for word in query if (word not in last_query)]
    total_unique_term = [word for word in query] + [word for word in last_query]
    return len(set(new_unique_term))/len(set(total_unique_term))

data['last_query'] = data['query'].shift(1)
data.loc[data['query_order']==1 , 'last_query'] = ''
data['new_unique_query_terms_num'] = data.apply(lambda r: new_unique_query_terms_num(r['query'], r['last_query']), axis=1)

"""
QuerySim
"""
def last_query_similarity(query, last_query):
    shared_unique_term = [word for word in query if (word in last_query)]
    total_unique_term = [word for word in query] + [word for word in last_query]
    return len(set(shared_unique_term))/len(set(total_unique_term))

data['last_query'] = data['query'].shift(1)
data.loc[data['query_order']==1 , 'last_query'] = ''
data['query_similarity'] = data.apply(lambda r: last_query_similarity(r['query'], r['last_query']), axis=1)

def anno_annotation_score(r, score_type='query_relevance'):
    condition = (all_relevance_annotation['url']==r['result_url'])&(all_relevance_annotation['task']==r['task_id'])
    try:
        return all_relevance_annotation.loc[condition, score_type].values[0]
    except:
        return

anno_annotation['query_relevance_score'] = anno_annotation.progress_apply(anno_annotation_score, args=('query_relevance',), axis=1)
anno_annotation['task_relevance_score'] = anno_annotation.progress_apply(anno_annotation_score, args=('task_relevance',), axis=1)

anno_annotation['query_relevance'] = anno_annotation['query_relevance_score'].apply(lambda r: 1 if r>=1 else 0)
anno_annotation['query_key'] = anno_annotation['query_relevance_score'].apply(lambda r: 1 if r>=2 else 0)
anno_annotation['task_relevance'] = anno_annotation['task_relevance_score'].apply(lambda r: 1 if r>=1 else 0)
anno_annotation['task_key'] = anno_annotation['task_relevance_score'].apply(lambda r: 1 if r>=2 else 0)

def ave_relevant_score(r, col):
    condition = (anno_annotation['studentID']==r['studentID'])& \
                (anno_annotation['task_id']==r['task_id'])& \
                (anno_annotation['query']==r['query'])
    return anno_annotation.loc[condition, col].mean()
data['ave_query_relevant_score'] = data.progress_apply(ave_relevant_score, args=('query_relevance_score', ), axis=1)
data['ave_task_relevant_score'] = data.progress_apply(ave_relevant_score, args=('task_relevance_score', ), axis=1)
data['ave_query_relevant_score'].fillna(0, inplace=True)
data['ave_task_relevant_score'].fillna(0, inplace=True)
exit_condition = ['ave_query_relevant_score', 'ave_task_relevant_score']

def get_click_precision(r, relevance_col):
    condition = (anno_annotation['studentID']==r['studentID'])& \
                (anno_annotation['task_id']==r['task_id'])& \
                (anno_annotation['query']==r['query'])
    help_df = anno_annotation.loc[condition, :]
    help_df.drop_duplicates(['result_id'])
    return help_df[relevance_col].sum()/len(help_df)
data['click_precision_query'] = data.progress_apply(get_click_precision, args=('query_relevance',), axis=1)
data['click_precision_task'] = data.progress_apply(get_click_precision, args=('task_relevance',), axis=1)
data['click_precision_query'].fillna(0, inplace=True)
data['click_precision_task'].fillna(0, inplace=True)

"""
number of click@ 3, 5, 10
"""


def NumOfClick(r, start, end):
    click_data = anno_annotation.loc[(anno_annotation['studentID'] == r['studentID']) &
                                     (anno_annotation['task_id'] == r['task_id']) &
                                     (anno_annotation['query'] == r['query']) &
                                     (anno_annotation['rank'] >= start) &
                                     (anno_annotation['rank'] <= end)]
    click_data.drop_duplicates(['result_url'], inplace=True)
    return len(click_data)


anno_annotation['rank'] = anno_annotation['result_id'].progress_apply(lambda r: int(r.split('_')[-1]))
data['num_of_click3'] = data.progress_apply(NumOfClick, axis=1, args=(1, 3,))
data['num_of_click5'] = data.progress_apply(NumOfClick, axis=1, args=(1, 5,))
data['num_of_click10'] = data.progress_apply(NumOfClick, axis=1, args=(6, 10,))

anno_annotation['url_rank'] = anno_annotation['result_id'].apply(lambda r: int(r.split('_')[1]))

def click_dep(r, col):
    condition = (anno_annotation['studentID']==r['studentID'])& \
                (anno_annotation['task_id']==r['task_id'])& \
                (anno_annotation['query']==r['query'])
    return anno_annotation.loc[condition, col].max()
data['click_depth'] = data.apply(click_dep, args=('url_rank',), axis=1)
data['click_depth'].fillna(data['click_depth'].max(), inplace=True)

"""
Cost-Benefit
"""
data['Query_Cost-Benefit-1'] = data.apply(lambda r: r['query_relevance_count']/r['clicked_doc_count'] if r['clicked_doc_count']!=0 else 0, axis=1)
data['Task_Cost-Benefit-1'] = data.apply(lambda r: r['task_relevance_count']/r['clicked_doc_count'] if r['clicked_doc_count']!=0 else 0, axis=1)

data['Query_Cost-Benefit-2'] = data.apply(lambda r: r['query_relevance_count']/r['click_depth'] if r['click_depth']!=0 else 0, axis=1)
data['Task_Cost-Benefit-2'] = data.apply(lambda r: r['task_relevance_count']/r['click_depth'] if r['click_depth']!=0 else 0, axis=1)

data['Query_Cost-Benefit-3'] = data['Query_Cost-Benefit-2']*data['SERPtime']
data['Task_Cost-Benefit-3'] = data['Task_Cost-Benefit-2']*data['SERPtime']

data = data.rename(columns={'SCROLL_count': 'ScrollDist',
                            'HOVER_count': 'HoverCount',
                            'clicked_doc_count': 'ClickCount',
                            'query_order': 'QueryOrder',
                            'query_len': 'QueryLength',
                            'query_dwell_time': 'QueryDwellTime',
                            'query_relevance_count': 'QueryRelDocCount',
                            'query_key_count': 'QueryKeyDocCount',
                            'task_relevance_count': 'TaskRelDocCount',
                            'task_key_count': 'TaskKeyDocCount',
                            'query_nDCG3': 'QueryNDCG@3',
                            'query_nDCG5': 'QueryNDCG@5',
                            'query_nDCG10': 'QueryNDCG@10',
                            'task_nDCG3': 'TaskNDCG@3',
                            'task_nDCG5': 'TaskNDCG@5',
                            'task_nDCG10': 'TaskNDCG@10',
                            'query_DCG3': 'QueryDCG@3',
                            'query_DCG5': 'QueryDCG@5',
                            'query_DCG10': 'QueryDCG@10',
                            'task_DCG3': 'TaskDCG@3',
                            'task_DCG5': 'TaskDCG@5',
                            'task_DCG10': 'TaskDCG@10',
                            'query_precision3': 'QueryPrecision@3',
                            'query_precision5': 'QueryPrecision@5',
                            'query_precision10': 'QueryPrecision@10',
                            'task_precision3': 'TaskPrecision@3',
                            'task_precision5': 'TaskPrecision@5',
                            'task_precision10': 'TaskPrecision@10',
                            'num_of_click3': 'Clicks@3',
                            'num_of_click5': 'Clicks@5',
                            'num_of_click10': 'Clicks@10',
                            'ave_query_relevant_score': 'AveQueryRelScore',
                            'ave_task_relevant_score': 'AveTaskRelScore',
                            'new_unique_query_terms_num': 'NewTerm',
                            'query_similarity': 'QuerySim',
                            'click_depth': 'ClickDepth',
                            'session_id': 'SessionID'})
data.to_csv('data/THU-KDD19/KDD19_total_feature.csv', index=False)