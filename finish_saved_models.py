"""Finish validation-selected final evaluation using already trained weights."""
from experiment_code import *

data=load_data()
y,s,tags,train,valid,pretest,test=data
p=OUT/'PMM_B_formal.json'
if p.exists():
    r=json.loads(p.read_text())
    progress=json.loads((OUT/'PMM_B_formal_progress.json').read_text())
    r.update(status='interrupted',observed_progress_steps=progress[-1]['step'],
        reason='Execution session ended during conversation interruption; no final checkpoint or validation forecast was saved.',test_evaluated=False)
    write_json(p,r)
selection={'selected':'PMM_A','phase':'formal','max_steps':2000,
    'selection_metric':'validation Overall pooled_scaled_crps among completed PMM runs only',
    'validation_scores':{'PMM_A':json.loads((OUT/'PMM_A_formal.json').read_text())['validation_scrps']},
    'excluded':{'PMM_B':'interrupted; no complete validation result','PMM_C':'not run within time budget'},
    'selected_at_epoch':time.time(),'test_evaluated':False,
    'limitation':'Only one PMM configuration completed; no claim of successful multi-setting tuning.'}
write_json(OUT/'selection.json',selection)

for name in ['GMM_fixed','PMM_A']:
    run_id=name+'_formal'
    r=json.loads((OUT/f'{run_id}.json').read_text())
    cls=GMM if name=='GMM_fixed' else PMM
    base=NHITS(h=12,input_size=24,loss=cls(n_components=10,quantiles=QUANTILES),
        hist_exog_list=['month'],max_steps=2000,scaler_type='robust',learning_rate=r['lr'],
        valid_loss=sCRPS(quantiles=QUANTILES),random_seed=1,
        accelerator='cpu',devices=1,logger=False,enable_progress_bar=False,enable_model_summary=False)
    base.load_state_dict(torch.load(OUT/f'{run_id}_weights.pt',weights_only=True))
    hint=HINT(h=12,S=s.to_numpy(),model=base,reconciliation='BottomUp',alias='NHITS')
    def predict(history, targets):
        ds,uids,*_=TimeSeriesDataset.from_df(history)
        assert list(uids)==list(s.index)
        np.random.seed(1)
        values=hint.predict(ds,random_seed=1)
        return pd.concat([targets[['unique_id','ds']].reset_index(drop=True),
            pd.DataFrame(values,columns=['NHITS'+c for c in base.loss.output_names])],axis=1)
    # Confirm restored model reproduces the saved validation output before test.
    reproduced=predict(train,valid)
    previous=pd.read_parquet(OUT/f'{run_id}_validation_predictions.parquet')
    cols=[c for c in reproduced if c not in ['unique_id','ds']]
    difference=float(np.abs(reproduced[cols].to_numpy()-previous[cols].to_numpy()).max())
    assert difference<1e-4,(name,difference)
    r['restored_validation_prediction_max_abs_diff']=difference
    pred=predict(pretest,test)
    pred.to_parquet(OUT/f'{run_id}_test_predictions.parquet')
    r['test_scrps']=score(pred,test,tags,pretest,run_id,'test')
    r['test_evaluated']=True
    write_json(OUT/f'{run_id}.json',r)
    print('FINAL_TEST',run_id,r['test_scrps'],flush=True)

# Original unmodified GMM is evaluated only after setting selection, separately.
raw=pd.read_parquet(OUT/'original_gmm_test_predictions.parquet').rename(columns=lambda c:c.replace('HINT','NHITS'))
raw_score=score(raw,test,tags,pretest,'original_gmm','test')
write_json(OUT/'original_gmm_test_summary.json',{'test_pooled_scrps':raw_score,
    'warning':'Original ordering mismatch preserved; not a fair corrected-model baseline.'})
selection['test_evaluated']=True
write_json(OUT/'selection.json',selection)
print('ORIGINAL_TEST',raw_score,flush=True)
