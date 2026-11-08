"""Source embedded verbatim in ＰＰＰ.ipynb; all outputs are local."""
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('MPLCONFIGDIR', str(Path('.mpl-cache').resolve()))

import json
import time
import hashlib
import traceback
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback
from datasetsforecast.hierarchical import HierarchicalData
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, HINT
from neuralforecast.losses.pytorch import GMM, PMM, sCRPS
from neuralforecast.utils import augment_calendar_df
from neuralforecast.tsdataset import TimeSeriesDataset
from hierarchicalforecast.evaluation import evaluate
from utilsforecast.losses import scaled_crps, mae, rmse

torch.set_num_threads(1)
OUT = Path('experiment_results')
OUT.mkdir(exist_ok=True)
HORIZON = 12
LEVELS = np.arange(2, 100, 2).tolist()
QUANTILES = np.arange(1, 100) / 100
SEED = 1

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=lambda x: float(x) if np.isscalar(x) else str(x)))
    temporary.replace(path)

def load_data():
    y, s, tags = HierarchicalData.load('./data', 'TourismLarge')
    y['ds'] = pd.to_datetime(y['ds'])
    y, _ = augment_calendar_df(df=y, freq='M')
    if 'unique_id' in s.columns:
        s = s.set_index('unique_id')
    # HINT BottomUp requires S's original order, with bottom identities last.
    # Merely sorting S alphabetically would break BottomUp's bottom-row assumption.
    raw_ids = TimeSeriesDataset.from_df(y)[1]
    raw_match = list(raw_ids) == list(s.index)
    dtype = pd.CategoricalDtype(categories=s.index.tolist(), ordered=True)
    y['unique_id'] = y['unique_id'].astype(dtype)
    y = y.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    assert list(TimeSeriesDataset.from_df(y)[1]) == list(s.index)
    assert np.allclose(s.iloc[-s.shape[1]:].to_numpy(), np.eye(s.shape[1]))
    assert not y.duplicated(['unique_id', 'ds']).any()
    assert np.isfinite(y['y']).all() and (y['y'] >= 0).all()
    # Preserve original real-valued observations: no rounding or unit conversion.
    test = y.groupby('unique_id', observed=True).tail(HORIZON).copy()
    pretest = y.drop(test.index).copy()
    valid = pretest.groupby('unique_id', observed=True).tail(HORIZON).copy()
    train = pretest.drop(valid.index).copy()
    for frame in [y, pretest, train, valid, test]:
        assert frame.groupby('unique_id', observed=True).size().nunique() == 1
    stats = train.groupby('unique_id', observed=True)['y'].agg(['mean', 'var'])
    audit = {
        'series': int(y.unique_id.nunique()), 'rows': len(y),
        'S_shape': list(s.shape), 'negative_count': int((y.y < 0).sum()),
        'fractional_count': int((np.abs(y.y - np.round(y.y)) > 1e-6).sum()),
        'fractional_fraction': float((np.abs(y.y - np.round(y.y)) > 1e-6).mean()),
        'zero_fraction_train': float((train.y == 0).mean()),
        'raw_NeuralForecast_order_matches_S': raw_match,
        'fixed_order_matches_S': True,
        'train_variance_over_mean_median': float((stats['var'] / stats['mean'].replace(0, np.nan)).median()),
        'variance_ratio_is_unconditional_not_proof_of_conditional_overdispersion': True,
        'splits': {name: {'start':str(df.ds.min().date()),'end':str(df.ds.max().date()),
                           'rows':len(df), 'months_per_series':len(df)//len(s)}
                   for name, df in [('train',train),('validation',valid),('test',test)]},
        'data_sha256': hashlib.sha256(Path('data/hierarchical/TourismLarge/data.csv').read_bytes()).hexdigest(),
    }
    # Check observed hierarchy using training data only.
    pivot = train.pivot(index='unique_id', columns='ds', values='y').reindex(s.index)
    audit['observed_hierarchy_max_abs_error_train'] = float(np.abs(pivot.to_numpy()-s.to_numpy()@pivot.reindex(s.columns).to_numpy()).max())
    write_json(OUT/'data_audit.json', audit)
    print('DATA_AUDIT', json.dumps(audit, ensure_ascii=False), flush=True)
    return y, s, tags, train, valid, pretest, test

class Progress(Callback):
    def __init__(self, name, deadline):
        self.name, self.deadline = name, deadline
        self.started = time.time()
        self.records = []
        self.timed_out = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 50 == 0:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            rec = {'step':trainer.global_step,'loss':float(loss.detach().cpu()),
                   'seconds':time.time()-self.started}
            self.records.append(rec)
            write_json(OUT/f'{self.name}_progress.json', self.records)
            print(self.name, rec, flush=True)
        if time.time() >= self.deadline:
            self.timed_out = True
            trainer.should_stop = True

    def on_validation_end(self, trainer, pl_module):
        # Local recovery checkpoint only; do not change optimization or restore
        # another epoch when comparing against the original final-weight policy.
        if not trainer.sanity_checking and trainer.global_step and trainer.global_step % 250 == 0:
            path = OUT / f'{self.name}_latest.ckpt'
            temporary = path.with_suffix('.ckpt.tmp')
            trainer.save_checkpoint(temporary)
            temporary.replace(path)
            write_json(OUT/f'{self.name}_checkpoint.json',
                       {'step':trainer.global_step,'file':str(path),'saved_epoch':time.time()})

def score(pred, truth, tags, history, name, phase):
    pred = pred.rename(columns=lambda c: c.replace('.0', ''))
    joined = truth[['unique_id','ds','y']].merge(pred, on=['unique_id','ds'], validate='one_to_one')
    assert len(joined) == len(truth)
    columns = [c for c in pred.columns if c not in ['unique_id','ds']]
    assert np.isfinite(joined[columns].to_numpy()).all()
    result = evaluate(df=joined, tags=tags, metrics=[scaled_crps, mae, rmse],
                      models=['NHITS'], level=LEVELS,
                      train_df=history[['unique_id','ds','y']])
    result.to_csv(OUT/f'{name}_{phase}_metrics.csv', index=False)
    diagnostics = []
    for level, ids in {**tags, 'Overall':np.concatenate(list(tags.values()))}.items():
        part = joined[joined.unique_id.isin(ids)]
        diagnostics.append({'level':level,
            'coverage90': float(((part.y>=part['NHITS-lo-90']) & (part.y<=part['NHITS-hi-90'])).mean()),
            'width90': float((part['NHITS-hi-90']-part['NHITS-lo-90']).mean()),
            'bias': float((part['NHITS']-part.y).mean()),
            'negative_mean_count':int((part['NHITS']<0).sum())})
    pd.DataFrame(diagnostics).to_csv(OUT/f'{name}_{phase}_diagnostics.csv',index=False)
    qs = [f'NHITS-lo-{lv}' for lv in reversed(LEVELS)] + ['NHITS-median'] + [f'NHITS-hi-{lv}' for lv in LEVELS]
    assert (np.diff(joined[qs].to_numpy(),axis=1)>=-1e-6).all()
    # Current utilsforecast averages per-series ratios. An all-zero truth series
    # divides by machine epsilon and can dominate the result (~1e13). Retain
    # that exact library metric above, and also compute the pooled definition
    # written in the original notebook. Use the same 98 interval quantiles.
    pooled_rows=[]
    eval_q=np.concatenate([np.arange(1,50),np.arange(51,100)])/100
    eval_cols=[c for c in qs if c!='NHITS-median']
    for level, ids in {**tags,'Overall':np.concatenate(list(tags.values()))}.items():
        part=joined[joined.unique_id.isin(ids)]
        err=part['y'].to_numpy()[:,None]-part[eval_cols].to_numpy()
        pinball=np.maximum(eval_q*err,(eval_q-1)*err)
        denom=float(part.y.abs().sum())
        pooled=float(2*pinball.sum()/(len(eval_q)*denom)) if denom>0 else None
        pooled_rows.append({'level':level,'metric':'pooled_scaled_crps','NHITS':pooled,
                            'zero_total_series':int((part.groupby('unique_id',observed=True).y.sum()==0).sum())})
    pd.DataFrame(pooled_rows).to_csv(OUT/f'{name}_{phase}_pooled_metrics.csv',index=False)
    return next(row['NHITS'] for row in pooled_rows if row['level']=='Overall')

def fit_run(config, steps, data, deadline, phase='short'):
    y, s, tags, train, valid, pretest, test = data
    name = config['name']
    run_id = f'{name}_{phase}'
    cb = Progress(run_id, deadline)
    loss_cls = GMM if config['loss']=='GMM' else PMM
    loss = loss_cls(n_components=config.get('components',10), quantiles=QUANTILES,
                    weighted=config.get('weighted',False), num_samples=1000)
    base = NHITS(h=12,input_size=24,loss=loss,hist_exog_list=['month'],
                 max_steps=steps,early_stop_patience_steps=10,val_check_steps=min(50,steps),
                 scaler_type='robust',learning_rate=config['lr'],
                 valid_loss=sCRPS(quantiles=QUANTILES),random_seed=SEED,
                 accelerator='cpu',devices=1,logger=False,enable_progress_bar=False,
                 enable_model_summary=False,callbacks=[cb])
    model = HINT(h=12,S=s.to_numpy(),model=base,reconciliation='BottomUp',alias='NHITS')
    nf = NeuralForecast(models=[model],freq='MS')
    record = {**config,'phase':phase,'max_steps':steps,'seed':SEED,
              'num_samples':1000,'input_size':24,'scaler':'robust',
              'val_check_steps':min(50,steps),'early_stop_patience_steps':10,
              'started_epoch':time.time(),'test_evaluated':False,
              'ordering_fixed':True,'checkpoint_policy':'final weights, matching original notebook'}
    write_json(OUT/f'{run_id}.json',record)
    try:
        nf.fit(df=pretest,val_size=12)
        trained=nf.models[0].model
        assert list(nf.uids)==list(s.index)
        record['actual_steps']=len(trained.train_trajectories)
        record['train_trajectories']=trained.train_trajectories
        record['valid_trajectories']=trained.valid_trajectories
        record['timed_out']=cb.timed_out
        record['status']='completed' if not cb.timed_out else 'time_limited'
        record['stop_reason']='time_limit' if cb.timed_out else (
            'max_steps' if record['actual_steps']>=steps else 'early_stopping')
        np.random.seed(SEED)
        pred=nf.predict(df=train,random_seed=SEED)
        pred.to_parquet(OUT/f'{run_id}_validation_predictions.parquet')
        record['validation_scrps']=score(pred,valid,tags,train,run_id,'validation')
        torch.save(trained.state_dict(),OUT/f'{run_id}_weights.pt')
        # Reconciled point means must match the summing matrix for the fixed order.
        pivot=pred.pivot(index='unique_id',columns='ds',values='NHITS').reindex(s.index)
        record['mean_coherence_max_abs_error']=float(np.abs(pivot.to_numpy()-s.to_numpy()@pivot.reindex(s.columns).to_numpy()).max())
        print('FINISHED',run_id,record['actual_steps'],record['validation_scrps'],flush=True)
    except Exception as exc:
        record.update(status='failed',error=repr(exc))
        traceback.print_exc()
        nf=None
    record['elapsed_seconds']=time.time()-record['started_epoch']
    write_json(OUT/f'{run_id}.json',record)
    return nf,record

def predict_saved_model(run_id, history, S, sampling_seed=None):
    """Reload a local fitted model and forecast 12 months without retraining.

    `history` contains unique_id/ds/y/month. S is indexed by unique_id, in its
    original hierarchy order. No evaluation targets are needed or accessed.
    """
    record=json.loads((OUT/f'{run_id}.json').read_text())
    if record.get('status')!='completed':
        raise ValueError(f'{run_id} is not a completed run')
    history=history.copy()
    if set(history.unique_id.astype(str))!=set(S.index.astype(str)):
        raise ValueError('History must contain every series in the hierarchy')
    history['unique_id']=history['unique_id'].astype(
        pd.CategoricalDtype(categories=S.index.tolist(),ordered=True))
    history=history.sort_values(['unique_id','ds'])
    loss_cls=GMM if record['loss']=='GMM' else PMM
    base=NHITS(h=12,input_size=24,
        loss=loss_cls(n_components=record.get('components',10),quantiles=QUANTILES,
                      weighted=record.get('weighted',False),num_samples=1000),
        hist_exog_list=['month'],max_steps=record['max_steps'],
        scaler_type='robust',learning_rate=record['lr'],valid_loss=sCRPS(quantiles=QUANTILES),
        random_seed=record['seed'],accelerator='cpu',devices=1,logger=False,
        enable_progress_bar=False,enable_model_summary=False)
    base.load_state_dict(torch.load(OUT/f'{run_id}_weights.pt',weights_only=True))
    hint=HINT(h=12,S=S.to_numpy(),model=base,reconciliation='BottomUp',alias='NHITS')
    dataset,uids,last_dates,*_=TimeSeriesDataset.from_df(history)
    assert list(uids)==list(S.index)
    inference_seed=record['seed'] if sampling_seed is None else sampling_seed
    np.random.seed(inference_seed)
    values=hint.predict(dataset,random_seed=inference_seed)
    dates=np.concatenate([pd.date_range(pd.Timestamp(date)+pd.offsets.MonthBegin(1),
                          periods=12,freq='MS').to_numpy() for date in last_dates])
    pred=pd.DataFrame({'unique_id':np.repeat(np.asarray(uids),12),'ds':dates})
    pred['unique_id']=pred['unique_id'].astype(history['unique_id'].dtype)
    return pd.concat([pred,pd.DataFrame(values,columns=['NHITS'+c for c in base.loss.output_names])],axis=1)

def run_experiment(steps=200, budget_seconds=600, phase='short', evaluate_test=False):
    data=load_data()
    # At most 3 PMM settings. No rounding, no test-dependent decisions.
    configs=[{'name':'GMM_fixed','loss':'GMM','lr':1e-3},
             {'name':'PMM_A','loss':'PMM','lr':1e-3},
             {'name':'PMM_B','loss':'PMM','lr':3e-4},
             {'name':'PMM_C','loss':'PMM','lr':3e-4,'weighted':True}]
    deadline=time.time()+budget_seconds
    runs={}
    for config in configs:
        if time.time()>deadline-45:
            break
        nf,rec=fit_run(config,steps,data,deadline-30,phase)
        if nf is not None:
            runs[config['name']]=(nf,rec)
    pmm=[(k,r) for k,(nf,r) in runs.items() if r['loss']=='PMM' and 'validation_scrps' in r]
    if not pmm or 'GMM_fixed' not in runs:
        write_json(OUT/'selection.json',{'status':'incomplete','phase':phase,'test_evaluated':False})
        return runs
    # Compare only equal budgets without timeout. Never compare an incomplete run
    # as if it had received the same training budget.
    pmm=[(k,r) for k,r in pmm if not r.get('timed_out')]
    if not pmm:
        return runs
    selected=min(pmm,key=lambda item:item[1]['validation_scrps'])[0]
    selection={'selected':selected,'phase':phase,'max_steps':steps,
               'selection_metric':'validation Overall pooled_scaled_crps after HINT BottomUp; native scaled_crps also retained',
               'validation_scores':{k:r['validation_scrps'] for k,r in pmm},
               'selected_at_epoch':time.time(),'test_evaluated':False}
    write_json(OUT/'selection.json',selection)
    if not evaluate_test:
        print('VALIDATION_ONLY', selection, flush=True)
        return runs
    # Only now use test targets, once for the fixed baseline and chosen PMM.
    y,s,tags,train,valid,pretest,test=data
    for name in ['GMM_fixed',selected]:
        nf,record=runs[name]
        np.random.seed(SEED)
        pred=nf.predict(df=pretest,random_seed=SEED)
        run_id=f'{name}_{phase}'
        pred.to_parquet(OUT/f'{run_id}_test_predictions.parquet')
        record['test_scrps']=score(pred,test,tags,pretest,run_id,'test')
        record['test_evaluated']=True
        write_json(OUT/f'{run_id}.json',record)
        print('FINAL_TEST',run_id,record['test_scrps'],flush=True)
    selection['test_evaluated']=True
    write_json(OUT/'selection.json',selection)
    return runs
