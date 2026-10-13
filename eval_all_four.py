import json,pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,average_precision_score
from ml_experiments.models.ml_models import get_models
main=pd.read_csv('results/Label_ILD/tuned_hyperparameters/gridsearch_results_auroc.csv')
pars={(r.model,int(r.seed)):json.loads(r.best_parameters) for _,r in main[(main.split=='scaffold')&main.model.isin(['RF','XGB','SVM','KNN'])].iterrows()}
root=Path('data/split_data/Label_ILD_all_four/scaffold');rows=[]
for model in ['RF','XGB','SVM','KNN']:
 for seed in range(5):
  tr=pd.read_csv(root/f'seed_{seed}/train.csv');te=pd.read_csv(root/f'seed_{seed}/test.csv');ytr=tr.pop('Label_ILD');yte=te.pop('Label_ILD');drop=[c for c in tr if c.lower() in ('smiles','smile')];tr=tr.drop(columns=drop);te=te.drop(columns=drop);pipe=clone(get_models(tr,random_state=seed)[model]['pipeline']);pipe.set_params(**pars[(model,seed)]);pipe.fit(tr,ytr);p=pipe.predict_proba(te)[:,list(pipe.classes_).index(1)];rows.append({'model':model,'seed':seed,'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'n_test':len(yte),'positive_test':int(yte.sum())})
d=pd.DataFrame(rows);d.to_csv('results/Label_ILD_all_four_scaffold_all_models.csv',index=False);print(d.to_string(index=False));print('\nMEAN');print(d.groupby('model')[['roc_auc','pr_auc']].mean())
