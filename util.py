import pandas as pd
import math
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import scipy.signal as signal
import glob
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import catboost as cbt
import gc

#可替换为相应目录
testa_files = glob.glob("/tcdata/hy_round2_testA_20200225/*.csv")
train_files = glob.glob("/tcdata/hy_round2_train_20200225/*.csv")
testa_files = pd.Series(testa_files)
train_files = pd.Series(train_files)
print(len(testa_files))
def read(file):
    d = pd.read_csv(file)
    return d
test = pd.concat(testa_files.map(read).values, axis = 0)
train = pd.concat(train_files.map(read).values, axis = 0)


# In[4]:


# train = pd.read_csv('new_train.zip')
# test = pd.read_csv('new_testb.zip')
#读数据
train.columns = ['id','x','y','sd','fx','time','label']
# test=test[['渔船ID','x','y','速度','方向','time']]
test.columns = ['id','x','y','sd','fx','time']

df = pd.concat([train,test],ignore_index=True)
#删除速度异常
df = df[df.sd<13]
df


# In[5]:


#平滑
def smooth(sig, method):
    if method == 0:
        return sig
    elif method == 1:
        return signal.medfilt(volume=sig, kernel_size=5)
    elif method == 2:
        return signal.savgol_filter(sig, 5, 3, 0)
    elif method == 3:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        return np.convolve(sig, kernel, mode='same')
    
    
def smooth_cols(group):
    cols = ['x','y','sd','fx']
    for col in cols:
        sig = group[col]
        group[col] = smooth(sig,1)
    return group
df = df.groupby('id').apply(smooth_cols)
df


# In[6]:


feat = pd.DataFrame()
d=df.drop_duplicates(['id'])
feat['id'] = d['id']
feat['label'] = d['label']
feat.index=feat['id']


# In[7]:


#终极版统计特征
def stat1(group):
    data = group
    #data = data.sort_values()[5:-5]
    c = []
    c.append(data.mode().values[0])
    c.append(data.max())
    c.append(data.min())
    c.append(data.mean())
    c.append(data.ptp())
    c.append(data.std())
    c.append(data.median())
    c.append(data.kurt())
    c.append(data.skew())
    c.append(np.mean(np.abs(data - data.mean()))) 
#     c[name + '_abs_mean'] = np.mean(np.abs(data))
    
    c.append(c[1] / c[0])
    c.append(c[2] / c[0])
    c.append(c[3] / c[0])
    c.append(c[6] / c[0])
#     c.append(data.max()-data.min())
    return c


# In[8]:


#删除异常值
# df=df[(df['y']>4400000)&(df['y']<7000000)]
# df=df[df['sd']<13]
df = df.reset_index(drop=True)
#针对报点过多的进行删除
tr = df.groupby('id').size().reset_index()
ycbd = list(tr['id'][tr[0]>1000])
for i in ycbd:
    qinde = df[df['id']==i].index[0]
    hinde = df[df['id']==i].index[-1]
    qhin = []
    for i in range(qinde,hinde,5):
        qhin.append(i)
    df = df.drop(qhin,0)
    
# df['x']=np.log(df['x'])
# df['y']=np.log(df['y'])
# sdafasd


# In[9]:


# def rate(df,feat):
#     df1 = df.groupby(['id']).agg({feat:'diff','time':'diff'})
#     df1[feat+'rate'] = abs(3600*df1[feat]/df1['time'].dt.total_seconds())
#     df1.reset_index(inplace=True)
#     df1.columns = ['id',feat,'time_delta',feat+'_rate']
#     return(df1[['id',feat+'_rate']])
# df['x']=df['x'].astype('int')
# df['y']=df['y'].astype('int')
#穿的位移情况
df['date'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
df[['x_diff','y_diff','date_diff']] = df.groupby(['id']).agg({'x':'diff','y':'diff','date':'diff'})
df['wy'] = pow(pow(df['y_diff'],2)+pow(df['x_diff'],2),0.5)
df['sd'][df['wy']==0]=0
df['wy_rate'] = abs(3600*df['wy']/df['date_diff'].dt.total_seconds())


# In[10]:


df['wy_rate'] = abs(3600*df['wy']/df['date_diff'].dt.total_seconds())
# feat['wy_rate_mean']=df.groupby(['id'])['wy_rate'].mean()
feat['wy_rate_max'] = df.groupby(['id'])['wy_rate'].max()
# feat['wy_rate_median']=df.groupby(['id'])['wy_rate'].median()
# feat


# In[11]:


df


# In[12]:


#针对xy提取特征
sx = df.groupby('id')['x'].apply(stat1)
sy = df.groupby('id')['y'].apply(stat1)
# swy = df.groupby('id')['wy'].apply(stat1)


# In[13]:


st = ['_'+ n for n in ['mode','max','min','mean','ptp','std','median','kurt','skew','mad','max_mode','min_mode','mean_mode','median_mode']]


for i in range(len(st)):
    feat['x'+st[i]] = sx.map(lambda x:x[i])
    feat['y'+st[i]] = sy.map(lambda x:x[i])
#     feat['wy'+st[i]] = swy.map(lambda x:x[i])

feat['x_max_x_min'] = feat['x_max'] - feat['x_min']
feat['y_max_y_min'] = feat['y_max'] - feat['y_min']
feat['y_max_x_min'] = feat['y_max'] - feat['x_min']
feat['x_max_y_min'] = feat['x_max'] - feat['y_min']
feat['slope'] = feat['y_max_y_min'] / np.where(feat['x_max_x_min']==0, 0.001, feat['x_max_x_min'])
feat['area'] = feat['x_max_x_min'] *feat['y_max_y_min']


# In[14]:


df = df.reset_index(drop=True)


# In[15]:


def stat2(group):
    data = group
    #data = data.sort_values()[5:-5]
    #data = data-data.median()
    c = []
    c.append(data.mode().values[0])
    c.append(data.max())
#     c.append(data.min())
    c.append(data.mean())
    c.append(data.ptp())
#     c.append(data.std())
#     c.append(data.median())
    c.append(data.kurt())
    c.append(data.skew())
    c.append(np.mean(np.abs(data - data.mean()))) 
#     c[name + '_abs_mean'] = np.mean(np.abs(data))
    
    
    return c

ss = df.groupby('id')['sd'].apply(stat2)
sf = df.groupby('id')['fx'].apply(stat2)

#判断是否在有风情况下作业
dff = df[df['sd']==0]
dff = dff.drop_duplicates(['id','x','y'])
dff = dff.groupby(['id']).size().reset_index()
dff.index = dff['id']
feat['likef'] = feat['id'].map(pd.Series(dff[0]))

# st = ['_'+ n for n in ['mode','max','min','mean','ptp','std','median','kurt','skew','mad']]
st = ['_'+ n for n in ['max','mean','ptp','kurt','skew','mad']]
for i in range(len(st)):
    feat['sd'+st[i]] = ss.map(lambda x:x[i])
    feat['fx'+st[i]] = sf.map(lambda x:x[i])


# In[16]:


df


# In[28]:

#提取白天的作业特征
def baiday(df0,df1,n):
  df0[n+'sd_mean'] = df0['id'].map(df1.groupby(['id'])['sd'].mean())
  df0[n+'fx_mean'] = df0['id'].map(df1.groupby(['id'])['fx'].mean())
  df0 = df0.drop_duplicates(['id'])
  df0 = df0.sort_values(['id'])
  df0 = df0.reset_index()
  return df0

def hour(df):
    df = df[5:7]
    return int(df)

def day(df):
    df = df[2:4]
    return int(df)

#统计出现最多的x y等
dxy = df.groupby(['id','x','y']).size().reset_index()
dxy1 = dxy.sort_values(['id',0],ascending=False)
dxy1 = dxy1.drop_duplicates(['id'])
dxy1.index = dxy1['id']
feat['x'] = feat['id'].map(dxy1['x'])
feat['y'] = feat['id'].map(dxy1['y'])
feat['x_y_cs'] = feat['id'].map(dxy1[0])


  #统计每个船速度为0 的个数
df0 = df[df['sd']==0]
dfs0 = df0.groupby(['id','sd']).size().reset_index()
dfs0.index = dfs0['id']
feat['速度_=0'] = feat['id'].map(dfs0[0])

df['day'] = df['time'].apply(day)
df['hour'] = df['time'].apply(hour)



lbl = LabelEncoder()
# dfd=df.drop_duplicates(['id']).reset_index()
# feat['day']=feat['id'].map(dfd['day'])

df['cx'] = df['id'].map(df['id'].value_counts())

df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str') # 经纬度连起来代表确切地理位置
df['x_y'] = lbl.fit_transform(df['x_y'].astype(str))
df['x_y_count'] = df['id'].map(df.groupby('id')['x_y'].nunique().to_dict())
df['x_y_count'] = df['x_y_count']
df0 = df.copy()
df0 = df0.drop_duplicates(['id'])
df0.index = feat['id']
# feat['x_y']=df0['x_y']
# feat['x_y_count']=df0['x_y_count']
# #构造白天的数据
bai = df[(df['hour']>=5)&(df['hour']<=19)]
df1 = baiday(df,bai,'bai_')
df1.index = feat['id']
feat['bai_sd_mean'] = df1['bai_sd_mean']
feat['bai_fx_mean'] = df1['bai_fx_mean']
feat['cx'] = df1['cx']
feat


# In[29]:


#分位数
def qua(group):
    data = group
    #data = data.sort_values()[5:-5]
    c = []
    c.append(data.quantile(.01))
    c.append(data.quantile(.05))
    c.append(data.quantile(.25))
    c.append(data.quantile(.75))
    c.append(data.quantile(.95))
    c.append(data.quantile(.99))
    c.append(data.quantile(.75)-data.quantile(.25))
    return c



st = ['_'+ n for n in ['01','05','25','75','95','99','75-25']]
for i in ['x','y','sd','fx']:
    ss = df.groupby('id')[i].apply(qua)
    for j in range(len(st)):
        feat[i+st[j]] = ss.map(lambda x:x[j])
        


# In[30]:



df['sd_bin'] = pd.cut(df.sd,[0,1,2,6,8,15],labels=False)

bins = [[] for i in range(5)]
#对速度进行分箱
def get_sdbin(group):
    for i in range(5):
        c = len(group[group['sd_bin']==i])
        bins[i].append(c)
df.groupby('id').apply(get_sdbin) 

for i in range(5):
    feat['sd_bin'+str(i+1)] = bins[i]
    
    


# In[31]:




bins = [[] for i in range(5)]

#每个速度分箱挡位所占的比例
def get_sdbinr(group):
    for i in range(5):
        c = len(group[group['sd_bin']==i])
        bins[i].append(c/len(group))

df.groupby('id').apply(get_sdbinr) 

for i in range(5):
    feat['sd_binr'+str(i+1)] = bins[i]
feat
# for i in range(5):
#     feat['sd_bin'+str(i+1)] = feat['sd_bin'+str(i+1)]/feat['cx']
# feat


# In[32]:


#单个样本的里的处理
#差分特征
df[['x_diff','y_diff','sd_diff','fx_diff']] = df.groupby('id').agg({
    'x':'diff',
    'y':'diff',
    'sd':'diff',
    'fx':'diff'
})



df['x/y'] = df['x']/(df['y']+0.0001)    
df['sd/x'] = df['sd']/(df['x']+0.0001)
df['sd/y'] = df['sd']/(df['y']+0.0001)
df['wy'] = pow(pow(df['y_diff'],2)+pow(df['x_diff'],2),0.5)

feat


# In[43]:


def stat3(group):
    data = group
    #data = data.sort_values()[5:-5]
    c = []
#     print(data.mode().values)

    if all(data.mode().values==None):
        c.append(-1)
    else:
        c.append(list(data.mode().values)[0])
    c.append(data.max())
    c.append(data.min())
    c.append(data.mean())
    c.append(data.ptp())
    c.append(data.std())
    c.append(data.median())
    return c


cols = ['x_diff','y_diff','sd_diff','fx_diff','x/y','sd/x','sd/y']

st = ['_'+ n for n in ['mode','max','min','mean','ptp','std','median']]
# st = ['_'+ n for n in ['mode','max','min','mean']]


for col in cols:
    ss = df.groupby('id')[col].apply(stat3)
    for i in range(len(st)):
        feat[col+st[i]] = ss.map(lambda x:x[i])
        


# In[44]:


feat


# In[45]:


fi = ['x','y','x/y_min', 'x/y_mode', 'x/y_max', 'x_mode','y_mode']
for i in range(len(fi)):
    for j in range(i+1,len(fi)):
#         feat[fi[i]+'-'+fi[j]] = feat[fi[i]]-feat[fi[j]]
        feat[fi[j]+'-'+fi[i]] = feat[fi[j]]-feat[fi[i]]
        feat[fi[i]+'+'+fi[j]] = feat[fi[i]]+feat[fi[j]]
        feat[fi[i]+'*'+fi[j]] = feat[fi[i]]*feat[fi[j]]
#         feat[fi[i]+'/'+fi[j]] = feat[fi[i]]/feat[fi[j]]


# In[46]:


# feat.drop(feat.columns[-42:],axis=1,inplace=True)


# In[47]:


feat


# In[48]:


fi = [f for f in feat.columns if f not in ['label','id']]

feat = feat.fillna(0)
feat.drop(feat[fi].columns[feat[fi].std() == 0],axis=1,inplace=True)
# fea = ['x/y_mode', 'x/y_min', 'x/y_max', 'sd_bin1', 'x_mode', 'y_mode',
#        'sd_binr1', 'sd_binr2', 'sd_binr3', 'fx_diff_std', 'fx_diff_mean',
#        'y_kurt', 'x_min', 'y_skew', 'sd_skew', 'x/y_median', 'x_skew',
#        'x/y_std', 'x/y_mean', 'fx_99', 'x_kurt', 'y_99', 'sd_bin3',
#        'sd_diff_std', 'fx_diff_max', 'fx_95', 'y_mean_mode', 'fx_skew',
#        'sd_diff_max', 'fx_diff_min', 'fx_mean', 'x_ptp', 'fx_diff_ptp', 'x_25',
#        'y_95', 'fx_75-25', 'x_max', 'sd_diff_ptp', 'sd/y_median', 'x_05',
#        'fx_75', 'y_min', 'x_max_mode', 'fx_std', 'x_mean_mode', 'x_01',
#        'x/y_ptp', 'sd_diff_min', 'sd/x_median', 'y_std', 'y_75', 'x_median',
#        'x_std', 'fx_mad', 'y_median', 'fx_max', 'y_ptp', 'x_mad', 'x_75-25',
#        'x_95', 'y_mad', 'fx_median', 'x_75', 'y_75-25', 'fx_ptp', 'y_25',
#        'y_mean', 'x_mean', 'sd_binr4', 'sd/y_max', 'sd_mad', 'y_01', 'x_99',
#        'sd_bin4', 'y_05', 'sd_95', 'sd_99', 'sd_mean', 'sd_std', 'sd_75-25',
#        'sd/x_mean', 'sd/y_ptp', 'fx_25', 'sd/y_std', 'sd/x_max', 'sd/x_std',
#        'sd/y_mean', 'sd/x_ptp', 'sd_75', 'sd_binr5', 'sd_bin5', 'sd_max',
#        'sd_ptp', 'sd_25', 'fx_mode', 'sd_median']
# len(fea)

# In[49]:


#重复特征清理
corr=feat.corr()
for i in corr.columns:
    for j in corr[corr[i]==1].index.values:
        if j !=i:
            print(j,corr[corr[i]==1].index)
            feat=feat.drop(j,1)
print(feat)


f = feat
dfr = f.iloc[:-len(testa_files)]
dft = f.iloc[-len(testa_files):]
X_test = dft[[col for col in dfr.columns if col not in ['label','id']]]
X = dfr[[col for col in dfr.columns if col not in ['label','id']]]
y = dfr['label']
y = y.map({'拖网':0,'围网':1,'刺网':2})

'''
#存储特征
tr_list = []
te_list = []


#随机森林
oof_rf = np.zeros((len(X),3))
pred_rf = np.zeros((len(X_test),3))
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
clf = RandomForestClassifier(n_estimators=300, max_depth=20,random_state=2019,verbose=0,n_jobs=4,oob_score=True)

for index, (trn_idx, val_idx) in enumerate(kf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    X_train,y_train = X.iloc[trn_idx],y.iloc[trn_idx]
    X_val,y_val = X.iloc[val_idx],y.iloc[val_idx]
    clf.fit(X_train,y_train)
    oof_rf[val_idx] = clf.predict_proba(X_val)
    r = np.argmax(clf.predict_proba(X_val),axis=1)
    pred_rf += clf.predict_proba(X_test)
    print(f1_score(y_val.values, r, average='macro'))

pred_rf /= 5 
oof_rf_final = np.argmax(oof_rf, axis=1)
print('---------f1--final--------')
print(f1_score(y.values, oof_rf_final, average='macro'))
tr = pd.DataFrame(oof_rf,columns=['rf1','rf2','rf3'])
te = pd.DataFrame(pred_rf,columns=['rf1','rf2','rf3'])
tr_list.append(tr)
te_list.append(te)
'''

#lgb
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
oof_lgb = np.zeros((len(X),3))
pred_lgb = np.zeros((len(X_test),3))
imp = pd.DataFrame()


def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      
    score_vail = f1_score(y_true=labels, y_pred=pred, average='macro')
    
    return 'f1_score', score_vail, True


lgb_param = {    'boosting_type': 'gbdt', 
    'objective': 'multiclassova', 
    'num_class':3,
    'learning_rate': 0.09, 
#     'num_leaves': 42,
         
    'max_depth':-1,   
     'subsample': 0.5, 
    'colsample_bytree': 0.5,
    'is_unbalance': 'true',
    'metric':'None'
    }
#score = []

for index, (trn_idx, val_idx) in enumerate(skf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    trn_data = lgb.Dataset(X.values[trn_idx], y.values[trn_idx])
    val_data = lgb.Dataset(X.values[val_idx], y.values[val_idx])
    num_round = 10000
    clf = lgb.train(lgb_param, trn_data, num_round, valid_sets = [val_data], feval=f1_score_vail,verbose_eval = 100, 
                    early_stopping_rounds = 600)
    oof_lgb[val_idx] = clf.predict(X.values[val_idx], num_iteration = clf.best_iteration)
    #f1s = f1_score(y.values[val_idx],np.argmax(oof_lgb[val_idx], axis=1),average='macro')
    imp['imp_'+str(index+1)] = clf.feature_importance()
    pred_lgb += clf.predict(X_test.values, num_iteration=clf.best_iteration)

pred_lgb /= 5
oof_lgb_final = np.argmax(oof_lgb, axis=1)
#imp.index = fi #X.columns
print('---------f1--final--------')
print(f1_score(y.values, oof_lgb_final, average='macro'))

'''
tr = pd.DataFrame(oof_lgb,columns=['lgb1','lgb2','lgb3'])
te = pd.DataFrame(pred_lgb,columns=['lgb1','lgb2','lgb3'])
tr_list.append(tr)
te_list.append(te)


#catboost
oof_cbt = np.zeros((len(X),3))
pred_cbt = np.zeros((len(X_test),3))
#imp_cbt = pd.DataFrame()

for index, (trn_idx, val_idx) in enumerate(skf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    X_train,y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val,y_val = X.iloc[val_idx], y.iloc[val_idx]
    num_round = 2000
    cbt_model = cbt.CatBoostClassifier(iterations=num_round, learning_rate=0.1, max_depth=5, 
                                       verbose=100, early_stopping_rounds=200, custom_metric='F1',
                                      loss_function='MultiClass')
    #设置模型参数，verbose表示每100个训练输出打印一次
    cbt_model.fit(X_train, y_train, eval_set=(X_val, y_val),plot=False,use_best_model=True) #训练五折分割后的训练集
    gc.collect() #垃圾清理，内存清理
    oof_cbt[val_idx] = cbt_model.predict_proba(X_val)
    pred_cbt += cbt_model.predict_proba(X_test)
    print(f1_score(y_val.values, np.argmax(oof_cbt[val_idx],axis=1), average='macro'))


    #imp_cbt['imp_'+str(index+1)] = cbt_model.get_feature_importance()
pred_cbt /= 5   
oof_cbt_final = np.argmax(oof_cbt, axis=1)
#imp_cbt.index = fi #X.columns
print('---------f1--final--------')
print(f1_score(y.values, oof_cbt_final, average='macro'))

tr = pd.DataFrame(oof_cbt,columns=['cbt1','cbt2','cbt3'])
te = pd.DataFrame(pred_cbt,columns=['cbt1','cbt2','cbt3'])
tr_list.append(tr)
te_list.append(te)

#xgboost
oof_xgb = np.zeros((len(X),3))
pred_xgb = np.zeros((len(X_test),3))
def f1_macro(preds,dtrain):
    label=dtrain.get_label()
    preds = np.argmax(preds,axis=1)
    f1=f1_score(label,preds,average='macro')
    return 'f1-score',-float(f1)

xgb_param = {"objective": 'multi:softprob',
                  "booster" : "gbtree",
                  "eta": 0.1,
                  "max_depth":7,#9
                  "subsample": 0.8,#0.85
                  'eval_metric':'mlogloss',#logloss
                  "colsample_bytree": 0.6,#0.7
                  "colsample_bylevel":0.6,#0.8
                  'tree_method':'auto',                                
                  "thread":4,
                  "seed": 2020,
                    'num_class':3,
                 'is_unbalance': True
                  }

for index, (trn_idx, val_idx) in enumerate(kf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    trn_data = xgb.DMatrix(X.iloc[trn_idx], y.iloc[trn_idx])
    val_data = xgb.DMatrix(X.iloc[val_idx], y.iloc[val_idx])
    xgb_te = xgb.DMatrix(X_test)
    watchlist = [(trn_data, 'train'), (val_data, 'eval')]
    xgb_model =xgb.train(xgb_param,
                 trn_data,
                feval=f1_macro,
                 num_boost_round = 1000,#1699,1126
                 evals =watchlist,
                 verbose_eval=20,
                 early_stopping_rounds=200
                        )
    oof_xgb[val_idx] = xgb_model.predict(val_data, ntree_limit=xgb_model.best_ntree_limit)
    f1_s = f1_score(y.iloc[val_idx], np.argmax(oof_xgb[val_idx],axis=1), average='macro')
    print('f1----{}'.format(f1_s))
    pred_xgb += xgb_model.predict(xgb_te, ntree_limit=xgb_model.best_ntree_limit)

pred_xgb /= 5
oof_xgb_final = np.argmax(oof_xgb, axis=1)
#imp.index = fi #X.columns
print('---------f1--final--------')
print(f1_score(y.values, oof_xgb_final, average='macro'))

tr = pd.DataFrame(oof_xgb,columns=['xgb1','xgb2','xgb3'])
te = pd.DataFrame(pred_xgb,columns=['xgb1','xgb2','xgb3'])

tr_list.append(tr)
te_list.append(te)


#end
X_test = pd.concat(te_list,axis=1)
X = pd.concat(tr_list,axis=1)

reg = LogisticRegression(multi_class="multinomial",solver="newton-cg",max_iter=10,C=0.5)
reg.fit(X,y)
pred_reg = reg.predict_proba(X_test)
'''

#最终使用lgb单模
#stacking 框架由于提交失误未得到验证
pred = np.argmax(pred_lgb, axis=1)
sub = pd.DataFrame()
sub['id'] = dft.id
sub['label'] = pred
sub['label'] = sub['label'].map({0:'拖网',1:'围网',2:'刺网'})
print('测试通过')
print(sub)
sub.to_csv('result.csv',index=None, header=False)
