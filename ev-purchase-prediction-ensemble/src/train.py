import os, gc, time, json, traceback, warnings
from reproducibility import environment_metadata, file_sha256
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
try:
    from catboost import CatBoostClassifier
    CAT_OK=True; CAT_IMPORT_ERR=''
except Exception as e:
    CAT_OK=False; CAT_IMPORT_ERR=repr(e)

ROOT=Path(os.getenv('KAGGLE_INPUT_ROOT', '/kaggle/input')); OUT=Path(os.getenv('OUTPUT_DIR', '/kaggle/working')); OUT.mkdir(parents=True, exist_ok=True)
TARGET='Will_Buy_EV'; ID='id'; NF=int(os.getenv('CV_FOLDS','5')); SEED=int(os.getenv('EXPERIMENT_SEED','42')); THREADS=max(1,min(8,os.cpu_count() or 4))

def find_data():
    req=['train.csv','test.csv','sample_submission.csv']; cand=[]
    for p in ROOT.rglob('train.csv'):
        d=p.parent
        if all((d/x).is_file() for x in req):
            score=100 if 'playground-series-s6e9' in str(d).lower() else 0
            score += 10 if (d/'train.csv').stat().st_size>(d/'test.csv').stat().st_size else 0
            cand.append((score,d))
    if not cand: raise FileNotFoundError('Attach playground-series-s6e9 data: train.csv, test.csv, sample_submission.csv')
    return sorted(cand,key=lambda z:(z[0],str(z[1])),reverse=True)[0][1]
D=find_data(); print('DATA_DIR',D)
tr=pd.read_csv(D/'train.csv'); te=pd.read_csv(D/'test.csv'); sub0=pd.read_csv(D/'sample_submission.csv')
run_metadata = environment_metadata()
run_metadata.update({
    'seed': SEED,
    'cv_folds': NF,
    'data_dir_name': D.name,
    'input_sha256': {name: file_sha256(D/name) for name in ['train.csv','test.csv','sample_submission.csv']},
    'catboost_available': CAT_OK,
})
assert TARGET in tr and ID in tr and ID in te
features=[c for c in tr if c not in [TARGET,ID]]
assert features==[c for c in te if c!=ID]
assert len(sub0)==len(te) and list(sub0.columns)==[ID,TARGET] and sub0[ID].equals(te[ID])
assert tr[ID].is_unique and te[ID].is_unique
raw=tr[TARGET]
y=(raw.astype(str).str.lower().map({'yes':1,'no':0}) if raw.dtype=='object' else raw).astype('int8').to_numpy()
print('shapes',tr.shape,te.shape,sub0.shape,'target_rate',y.mean())
run_metadata.update({'train_shape': list(tr.shape), 'test_shape': list(te.shape), 'submission_shape': list(sub0.shape), 'target_rate': float(y.mean())})
(OUT/'run_metadata.json').write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding='utf-8')
print('missing',int(tr.isna().sum().sum()),int(te.isna().sum().sum()))

# Synthetic-pattern and domain features. No labels are used here.
def engineer(df):
    d=df[features].copy()
    inc='Annual_Income_USD'; com='Daily_Commute_km'; age='Age'; h='Charging_Stations_Near_Home'; w='Charging_Stations_Near_Work'
    if inc in d:
        d['income_log']=np.log1p(d[inc].clip(lower=0)); d['income_k']=(d[inc]//1000).astype('int32'); d['income_mod1000']=(d[inc]%1000).astype('int16'); d['income_floor30000']=(d[inc]==30000).astype('int8')
        for q in [100,250,500,1000,2500,5000,10000]: d[f'income_round_{q}']=(d[inc]/q).round().astype('int32')
    if com in d:
        d['commute_log']=np.log1p(d[com].clip(lower=0)); d['commute_round1']=d[com].round().astype('int16'); d['commute_x10_round']=(d[com]*10).round().astype('int16'); d['commute_floor5']=(d[com]==5).astype('int8')
        for q in [1,2,5,10]: d[f'commute_bin_{q}']=(d[com]//q).astype('int16')
    if inc in d and age in d: d['income_per_age']=d[inc]/(d[age]+1)
    if h in d and w in d:
        d['charging_total']=d[h]+d[w]; d['charging_min']=d[[h,w]].min(axis=1); d['charging_max']=d[[h,w]].max(axis=1); d['charging_gap']=d[w]-d[h]; d['charging_ratio']=(d[w]+1)/(d[h]+1)
    if com in d and h in d: d['commute_per_home_station']=d[com]/(d[h]+1)
    if com in d and w in d: d['commute_per_work_station']=d[com]/(d[w]+1)
    combos=[['Subsidy_Available','Environmental_Concern_Level'],['Home_Charging_Possible','Range_Anxiety_Level'],['City_Type','Range_Anxiety_Level'],['Current_Car_Type','City_Type'],['Subsidy_Available','Home_Charging_Possible'],['Environmental_Concern_Level','Range_Anxiety_Level']]
    for cs in combos:
        if all(c in d for c in cs): d['combo_'+'_'.join(cs)]=d[cs].astype(str).agg('|'.join,axis=1)
    return d
X=engineer(tr); T=engineer(te)
base_cat=X.select_dtypes(include=['object','category','bool']).columns.tolist()
base_num=[c for c in X if c not in base_cat]
folds=list(StratifiedKFold(NF,shuffle=True,random_state=SEED).split(X,y))

def prep_native(a,b,t):
    a=a.copy(); b=b.copy(); t=t.copy()
    cats=a.select_dtypes(include=['object','category','bool']).columns.tolist()
    for c in cats:
        lv=pd.Index(a[c].fillna('__MISSING__').astype(str).unique())
        a[c]=pd.Categorical(a[c].fillna('__MISSING__').astype(str),categories=lv)
        b[c]=pd.Categorical(b[c].fillna('__MISSING__').astype(str),categories=lv)
        t[c]=pd.Categorical(t[c].fillna('__MISSING__').astype(str),categories=lv)
    for c in [z for z in a if z not in cats]:
        med=a[c].median(); a[c]=a[c].fillna(med); b[c]=b[c].fillna(med); t[c]=t[c].fillna(med)
    return a,b,t,cats

def add_fold_te(a,b,t,ya):
    # Leakage-safe nested cross-fitting for outer-training rows.
    # Outer validation/test mappings use all outer-training rows only.
    a=a.copy(); b=b.copy(); t=t.copy(); ya=np.asarray(ya); smooth=40.0
    tecols=[c for c in a if (a[c].dtype=='object' or str(a[c].dtype).startswith('category') or c.startswith('income_round_') or c.startswith('commute_bin_'))]
    inner=StratifiedKFold(5,shuffle=True,random_state=SEED+917)
    for c in tecols:
        keya=a[c].fillna('__MISSING__').astype(str).reset_index(drop=True)
        keyb=b[c].fillna('__MISSING__').astype(str)
        keyt=t[c].fillna('__MISSING__').astype(str)
        train_te=np.zeros(len(a),dtype='float32')
        for inn_tr,inn_va in inner.split(np.zeros(len(ya)),ya):
            prior=float(ya[inn_tr].mean())
            stat=pd.DataFrame({'k':keya.iloc[inn_tr].to_numpy(),'y':ya[inn_tr]}).groupby('k')['y'].agg(['mean','count'])
            mp=((stat['mean']*stat['count']+prior*smooth)/(stat['count']+smooth)).to_dict()
            train_te[inn_va]=keya.iloc[inn_va].map(mp).fillna(prior).to_numpy(dtype='float32')
        prior=float(ya.mean())
        stat=pd.DataFrame({'k':keya.to_numpy(),'y':ya}).groupby('k')['y'].agg(['mean','count'])
        mp=((stat['mean']*stat['count']+prior*smooth)/(stat['count']+smooth)).to_dict()
        a[c+'_te']=train_te
        b[c+'_te']=keyb.map(mp).fillna(prior).to_numpy(dtype='float32')
        t[c+'_te']=keyt.map(mp).fillna(prior).to_numpy(dtype='float32')
    return a,b,t

def train_lgb(name,use_te,params):
    o=np.zeros(len(X)); p=np.zeros(len(T)); aucs=[]; started=time.time()
    for k,(ii,jj) in enumerate(folds,1):
        a=X.iloc[ii].copy(); b=X.iloc[jj].copy(); t=T.copy()
        if use_te: a,b,t=add_fold_te(a,b,t,y[ii])
        a,b,t,cats=prep_native(a,b,t)
        lgb_params=dict(objective='binary',n_estimators=1800,learning_rate=.03,num_leaves=31,min_child_samples=80,subsample=.85,colsample_bytree=.85,reg_alpha=.05,reg_lambda=1.5,random_state=SEED+k,n_jobs=THREADS,verbosity=-1,force_col_wise=True)
        lgb_params.update(params)
        m=LGBMClassifier(**lgb_params)
        m.fit(a,y[ii],eval_set=[(b,y[jj])],eval_metric='auc',categorical_feature=cats,callbacks=[early_stopping(100,verbose=False),log_evaluation(0)])
        o[jj]=m.predict_proba(b,num_iteration=m.best_iteration_)[:,1]; p+=m.predict_proba(t,num_iteration=m.best_iteration_)[:,1]/NF
        auc=roc_auc_score(y[jj],o[jj]); aucs.append(float(auc)); print(name,'fold',k,'AUC',f'{auc:.9f}','iter',m.best_iteration_)
        del a,b,t,m; gc.collect()
    full=float(roc_auc_score(y,o)); print(name,'OOF',f'{full:.9f}','seconds',time.time()-started)
    return {'name':name,'oof':o,'test':p,'fold_auc':aucs,'oof_auc':full,'seconds':time.time()-started}

results=[]; failures=[]
for name,use_te,params in [('lgb_native',False,{}),('lgb_fold_te',True,{'num_leaves':24,'min_child_samples':100,'reg_lambda':2.2})]:
    try: results.append(train_lgb(name,use_te,params))
    except Exception as e: failures.append(name+': '+traceback.format_exc()); print(failures[-1])
if not results: raise RuntimeError('All required LightGBM models failed')

# Optional CatBoost for diversity. Failure never stops the main flow.
if CAT_OK:
    try:
        o=np.zeros(len(X)); p=np.zeros(len(T)); aucs=[]; start=time.time(); cats=X.select_dtypes(include=['object','category','bool']).columns.tolist()
        for k,(ii,jj) in enumerate(folds,1):
            a=X.iloc[ii].copy(); b=X.iloc[jj].copy(); t=T.copy()
            for c in cats: a[c]=a[c].fillna('__MISSING__').astype(str); b[c]=b[c].fillna('__MISSING__').astype(str); t[c]=t[c].fillna('__MISSING__').astype(str)
            for c in [z for z in X if z not in cats]:
                med=a[c].median(); a[c]=a[c].fillna(med); b[c]=b[c].fillna(med); t[c]=t[c].fillna(med)
            m=CatBoostClassifier(iterations=1200,depth=7,learning_rate=.04,loss_function='Logloss',eval_metric='AUC',random_seed=SEED+k,l2_leaf_reg=6,random_strength=.5,thread_count=THREADS,verbose=False,allow_writing_files=False,task_type='CPU')
            m.fit(a,y[ii],cat_features=cats,eval_set=(b,y[jj]),early_stopping_rounds=80,verbose=False)
            o[jj]=m.predict_proba(b)[:,1]; p+=m.predict_proba(t)[:,1]/NF
            auc=roc_auc_score(y[jj],o[jj]); aucs.append(float(auc)); print('catboost fold',k,'AUC',f'{auc:.9f}')
            del a,b,t,m; gc.collect()
        results.append({'name':'catboost','oof':o,'test':p,'fold_auc':aucs,'oof_auc':float(roc_auc_score(y,o)),'seconds':time.time()-start})
        print('catboost OOF',results[-1]['oof_auc'])
    except Exception: failures.append('catboost: '+traceback.format_exc()); print(failures[-1])
else: failures.append('catboost import: '+CAT_IMPORT_ERR)

# OOF-only candidates: singles, pairwise and 3-model simplex with 0.1 steps, plus equal rank blends.
candidates=[]
for r in results: candidates.append((r['name'],r['oof'],r['test'],{r['name']:1.0}))
if len(results)>=2:
    for i in range(len(results)):
        for j in range(i+1,len(results)):
            for wi in np.arange(0,1.01,.1):
                wj=1-wi; candidates.append((f'prob_{results[i]["name"]}_{wi:.1f}_{results[j]["name"]}_{wj:.1f}',wi*results[i]['oof']+wj*results[j]['oof'],wi*results[i]['test']+wj*results[j]['test'],{results[i]['name']:float(wi),results[j]['name']:float(wj)}))
            ro=.5*pd.Series(results[i]['oof']).rank(pct=True).to_numpy()+.5*pd.Series(results[j]['oof']).rank(pct=True).to_numpy()
            rt=.5*pd.Series(results[i]['test']).rank(pct=True).to_numpy()+.5*pd.Series(results[j]['test']).rank(pct=True).to_numpy()
            candidates.append((f'rank_{results[i]["name"]}_{results[j]["name"]}',ro,rt,{results[i]['name']+'_rank':.5,results[j]['name']+'_rank':.5}))
if len(results)>=3:
    for a in range(11):
        for b in range(11-a):
            c=10-a-b; w=np.array([a,b,c])/10
            oo=sum(w[z]*results[z]['oof'] for z in range(3)); tt=sum(w[z]*results[z]['test'] for z in range(3))
            candidates.append((f'prob3_{a}_{b}_{c}',oo,tt,{results[z]['name']:float(w[z]) for z in range(3)}))
scored=[(float(roc_auc_score(y,o)),n,o,t,w) for n,o,t,w in candidates]
scored.sort(key=lambda z:z[0],reverse=True); best_auc,best_name,best_oof,best_test,best_w=scored[0]
print('TOP OOF CANDIDATES'); [print(x[1],f'{x[0]:.9f}',x[4]) for x in scored[:15]]

# Stable backup is successful single model with lowest fold std.
safe=min(results,key=lambda r:np.std(r['fold_auc']))
def save(name,p):
    s=sub0.copy(); p=np.asarray(p,float)
    assert len(p)==len(te) and np.isfinite(p).all(); s[TARGET]=np.clip(p,0,1); s.to_csv(OUT/name,index=False)
    return OUT/name
paths=[]; paths.append(save('submission_best.csv',best_test)); paths.append(save('submission_safe.csv',safe['test']))
for r in results: paths.append(save('submission_'+r['name']+'.csv',r['test']))
# Save best rank blend if available
ranked=[x for x in scored if x[1].startswith('rank_')]
if ranked: paths.append(save('submission_rank_blend.csv',ranked[0][3]))

oofdf=pd.DataFrame({ID:tr[ID],'target':y,'best_oof':best_oof})
for r in results: oofdf[r['name']+'_oof']=r['oof']
oofdf.to_csv(OUT/'oof_predictions.csv',index=False); paths.append(OUT/'oof_predictions.csv')
rows=[{'model':r['name'],'fold_auc':'|'.join(f'{v:.9f}' for v in r['fold_auc']),'oof_auc':r['oof_auc'],'fold_std':float(np.std(r['fold_auc'])),'seconds':r['seconds']} for r in results]
rows += [{'model':x[1],'fold_auc':'','oof_auc':x[0],'fold_std':np.nan,'seconds':0,'weights':json.dumps(x[4])} for x in scored[:20]]
pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False); paths.append(OUT/'model_comparison.csv')

# Hard validation after re-read.
ck=pd.read_csv(OUT/'submission_best.csv'); errs=[]
if ck.shape!=(len(te),len(sub0.columns)): errs.append('shape')
if ck.columns.tolist()!=sub0.columns.tolist(): errs.append('columns')
if not ck[ID].equals(sub0[ID]): errs.append('id/order')
p=ck[TARGET].to_numpy()
if not np.isfinite(p).all(): errs.append('nan/inf')
if not ((p>=0)&(p<=1)).all(): errs.append('range')
if np.std(p)<=0 or len(np.unique(p))<100: errs.append('constant/unique')
if any(str(c).startswith('Unnamed') for c in ck): errs.append('index column')
if errs:
    (OUT/'submission_best.csv').unlink(missing_ok=True); raise AssertionError('SUBMISSION VALIDATION FAILED: '+','.join(errs))
summary=f"""TOP-10-ORIENTED EV PIPELINE
Data: {D}
Train/test: {tr.shape} / {te.shape}
Target rate: {y.mean()}
Models: {[(r['name'],r['fold_auc'],r['oof_auc']) for r in results]}
Best: {best_name}
Best weights: {best_w}
Best OOF AUC: {best_auc}
Safe model: {safe['name']}
Failures: {failures or ['None']}
Leakage audit: ID excluded; no test target; all target encoding fitted inside each outer training fold; validation rows excluded from target statistics; blends selected only by OOF AUC.
SUBMISSION VALIDATION PASSED
Path: {OUT/'submission_best.csv'}
Shape: {ck.shape}
Min/max/mean/std/unique: {p.min()} / {p.max()} / {p.mean()} / {p.std()} / {len(np.unique(p))}
Files: {[str(x) for x in paths]}"""
(OUT/'run_summary.txt').write_text(summary,encoding='utf-8')
print('\nSUBMISSION VALIDATION PASSED\n',summary); print(ck.head())
