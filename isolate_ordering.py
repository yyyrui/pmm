"""Reuse original GMM weights and change only inference ordering; validation only."""
from experiment_code import *

y,s,tags,train,valid,pretest,test=load_data()
base=NHITS(h=12,input_size=24,loss=GMM(n_components=10,quantiles=QUANTILES),
           hist_exog_list=['month'],max_steps=2000,early_stop_patience_steps=10,
           val_check_steps=50,scaler_type='robust',learning_rate=1e-3,
           valid_loss=sCRPS(quantiles=QUANTILES),accelerator='cpu',devices=1,
           logger=False,enable_progress_bar=False,enable_model_summary=False)
base.load_state_dict(torch.load(OUT/'original_gmm_weights.pt',weights_only=True))
hint=HINT(h=12,S=s.to_numpy(),model=base,reconciliation='BottomUp',alias='NHITS')
dataset,uids,*_=TimeSeriesDataset.from_df(train)
assert list(uids)==list(s.index)
np.random.seed(1)
values=hint.predict(dataset,random_seed=1)
pred=valid[['unique_id','ds']].reset_index(drop=True).copy()
pred=pd.concat([pred,pd.DataFrame(values,columns=['NHITS'+c for c in base.loss.output_names])],axis=1)
pred.to_parquet(OUT/'original_gmm_order_only_validation_predictions.parquet')
value=score(pred,valid,tags,train,'original_gmm_order_only','validation')
write_json(OUT/'ordering_isolation.json',{'same_original_weights':True,'retrained':False,
    'only_inference_order_fixed':True,'validation_pooled_scrps':value,'test_evaluated':False,
    'sampling_seed':1})
print('ORDER_ONLY',value,flush=True)
