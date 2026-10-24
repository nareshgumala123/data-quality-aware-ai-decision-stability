"""Recompute reported values from saved case-level outputs; regenerate two figures.
Run from a directory containing results/ and source/research.py.
This script never retrains or changes model predictions.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,average_precision_score,f1_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results'); O=Path('final'); (O/'figures').mkdir(parents=True,exist_ok=True);(O/'tables').mkdir(exist_ok=True);(O/'supplement').mkdir(exist_ok=True)
m=json.loads((R/'run_metadata.json').read_text()); a=np.load(R/'case_audit.npz'); raw=pd.read_csv(R/'all_metrics.csv');clean=pd.read_csv(R/'clean_metrics.csv')
assert m['execution_complete'] and not m['diagnostic_random_only'] and m['repetitions']==30
assert len(raw)==3840
assert m['source_sha256']==hashlib.sha256(Path('source/research.py').read_bytes()).hexdigest()
assert set(raw['split'])=={'random','chronological'}
assert not raw.duplicated(['split','model','mechanism','rate','repetition','policy']).any()
assert raw[['accuracy','balanced_accuracy','roc_auc','average_precision','f1','brier','ece10','dfr','dss','harmful','beneficial']].notna().all().all()
np.testing.assert_allclose(raw.dfr,raw.harmful+raw.beneficial,atol=1e-14)
np.testing.assert_allclose(raw.dss+raw.dfr,1,atol=1e-14)
for split in ['random','chronological']:
 tr=a[split+'_train_row_ids'];va=a[split+'_validation_row_ids'];te=a[split+'_test_row_ids']
 assert len(set(tr)&set(va))==len(set(tr)&set(te))==len(set(va)&set(te))==0
 assert len(np.unique(np.concatenate([tr,va,te])))==41188
 assert [len(tr),len(va),len(te)]==[24712,8238,8238]
checks=[]; keys={'Logistic Regression':'lr','Random Forest':'rf','HistGradientBoosting':'hgb','Constant baseline':'constant'}
for row in clean.itertuples():
 y=a['target'][a[row.split+'_test_row_ids']];p=a[f'{row.split}_{keys[row.model]}_clean_probability']; d=p>=row.threshold
 values={'accuracy':accuracy_score(y,d),'balanced_accuracy':balanced_accuracy_score(y,d),'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'f1':f1_score(y,d,zero_division=0),'brier':brier_score_loss(y,p),'action_rate':d.mean()}
 for metric,value in values.items():np.testing.assert_allclose(value,getattr(row,metric),atol=1e-12)
 checks.append({'split':row.split,'model':row.model,'policy':row.policy,'clean_metrics_recomputed':True})
for group,g in raw.groupby(['split','model','mechanism','rate','policy']):
 split,model,kind,rate,policy=group
 assert len(g)==30 and set(g.repetition)==set(range(30))
 base=clean.query('split==@split and model==@model and policy==@policy').iloc[0]
 np.testing.assert_allclose(g.accuracy-base.accuracy,g.beneficial-g.harmful,atol=1e-13)
 np.testing.assert_allclose(g.balanced_accuracy-base.balanced_accuracy,g.delta_balanced_accuracy,atol=1e-13)
 np.testing.assert_allclose(g.roc_auc-base.roc_auc,g.delta_roc_auc,atol=1e-13)
 # Paired corruption exposure must be identical across models in each condition.
 if policy=='validation':
  y=a['target'][a[split+'_test_row_ids']];err=a[f'{split}_{keys[model]}_clean_error'];freq=a[f'{split}_{keys[model]}_{kind}_{int(round(rate*100))}_flip_frequency'].astype(float)
  np.testing.assert_allclose(freq.mean(),g.dfr.mean(),atol=3e-8)
  np.testing.assert_allclose(np.mean(freq*~err),g.harmful.mean(),atol=3e-8)
  np.testing.assert_allclose(np.mean(freq*err),g.beneficial.mean(),atol=3e-8)
  delta=.5*sum(np.mean((2*err.astype(int)-1)[y==c]*freq[y==c]) for c in [0,1])
  np.testing.assert_allclose(delta,g.delta_balanced_accuracy.mean(),atol=3e-8)
for cols,g in raw.groupby(['split','mechanism','rate','repetition']):
 assert g.corruption_seed.nunique()==1
 assert g.attempted_cell_fraction.nunique()==g.realized_cell_fraction.nunique()==1
s=raw.query("policy=='validation'").groupby(['split','model','mechanism','rate']).mean(numeric_only=True).reset_index()
s.to_csv(O/'supplement/all_condition_means.csv',index=False)
t1=clean.query("policy=='validation'")[['split','model','threshold','accuracy','balanced_accuracy','roc_auc','average_precision','f1','brier','ece10','action_rate']]
t1.to_csv(O/'tables/table1.csv',index=False)
t2=s.query("rate==.3 and model!='Constant baseline'").copy()
primary=pd.read_csv(R/'table2.csv')
for n in ['dfr_ci_low','dfr_ci_high']: t2[n]=np.nan
for idx,row in t2.query("split=='random'").iterrows():
 p=primary.query('model==@row.model and mechanism==@row.mechanism').iloc[0]
 t2.loc[idx,['dfr_ci_low','dfr_ci_high']]=p[['dfr_ci_low','dfr_ci_high']].to_numpy()
t2.to_csv(O/'tables/table2.csv',index=False)
# Figure 1: comparable corruption curves across both partition schemes.
colors={'Logistic Regression':'#0072B2','Random Forest':'#D55E00','HistGradientBoosting':'#009E73','Constant baseline':'#777777'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,axs=plt.subplots(2,2,figsize=(7.2,4.7),sharex=True,sharey=True)
summary=pd.read_csv(R/'primary_summary.csv')
for i,split in enumerate(['random','chronological']):
 for j,kind in enumerate(['missing','noise']):
  ax=axs[i,j]
  for model in keys:
   g=s.query('split==@split and mechanism==@kind and model==@model').sort_values('rate')
   ax.plot(100*g.rate,100*g.dfr,marker='o',ms=3,lw=1.4,color=colors[model],label=model,ls='--' if model=='Constant baseline' else '-')
   if split=='random':
    ci=summary.query('model==@model and mechanism==@kind').sort_values('rate')
    ax.fill_between(ci.rate.to_numpy()*100,ci.dfr_ci_low.to_numpy()*100,ci.dfr_ci_high.to_numpy()*100,color=colors[model],alpha=.14,linewidth=0)
  ax.set_title(f"{'abcd'[i*2+j]}  {split.capitalize()} / {'missingness' if kind=='missing' else 'noise'}",fontsize=9)
  ax.set_xticks([5,10,20,30]);ax.set_ylim(-.5,41);ax.grid(alpha=.18)
  if i==1:ax.set_xlabel('Nominal selected-cell rate (%)')
  if j==0:ax.set_ylabel('Decision flip rate (%)')
handles,labels=axs[0,0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.005),ncol=2,frameon=False,fontsize=8)
f.tight_layout(rect=[0,.09,1,1])
for ext in ['pdf','png']:f.savefig(O/'figures'/('Figure_1.'+ext),dpi=600,bbox_inches='tight')
plt.close(f)
# Figure 2: same conditions, directional action turnover and net metric changes.
g=t2.query("split=='random'").set_index(['model','mechanism']).loc[[(n,k) for n in list(keys)[:3] for k in ['missing','noise']]].reset_index()
x=np.arange(6);lab=[keys[r.model].upper()+'\n'+('Missing' if r.mechanism=='missing' else 'Noise') for r in g.itertuples()]
f,ax=plt.subplots(1,2,figsize=(7.2,2.8))
ax[0].bar(x,100*g.harmful,color='#D55E00',label='Harmful');ax[0].bar(x,100*g.beneficial,bottom=100*g.harmful,color='#009E73',label='Beneficial')
ax[0].set_title('a  Directional decision changes',fontsize=9);ax[0].set_ylabel('Fraction of all test cases (%)');ax[0].legend(fontsize=7,frameon=False)
ax[1].bar(x-.18,100*g.delta_balanced_accuracy,.36,color='#0072B2',label='Balanced accuracy');ax[1].bar(x+.18,100*g.delta_roc_auc,.36,color='#E69F00',label='ROC-AUC');ax[1].axhline(0,color='black',lw=.6)
ax[1].set_title('b  Changes in aggregate performance',fontsize=9);ax[1].set_ylabel('Corrupted minus reference (pp)');ax[1].legend(fontsize=7,frameon=False,loc='lower right')
for a0 in ax:a0.set_xticks(x,lab,fontsize=7);a0.grid(axis='y',alpha=.15);a0.set_axisbelow(True)
f.tight_layout(w_pad=1.2)
for ext in ['pdf','png']:f.savefig(O/'figures'/('Figure_2.'+ext),dpi=600,bbox_inches='tight')
plt.close(f)
reproduction=O/'supplement/reproduction_audit.json'
retraining=json.loads(reproduction.read_text())['status'] if reproduction.exists() else 'not supplied; this audit does not retrain'
report={'input_source_hash_matches':True,'dataset_rows':m['rows'],'evaluation_rows':len(raw),'clean_policy_rows_recomputed':len(checks),'condition_groups_recomputed':64,'full_condition_policy_groups':128,'repetitions_per_group':30,'partition_disjointness_checked':True,'flip_decomposition_checked':True,'accuracy_change_identity_checked':True,'balanced_accuracy_change_recomputed_from_cases':True,'paired_corruption_exposure_checked':True,'original_results_unchanged':True,'numerical_tolerance_case_frequency':3e-8,'figures':2,'tables':2,'independent_retraining':retraining,'journal_submission_ready':False}
(O/'supplement/numerical_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
