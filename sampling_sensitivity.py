"""Post-selection Monte Carlo sensitivity; fixed trained weights, no retuning."""
from experiment_code import *

selection=json.loads((OUT/'selection.json').read_text())
assert selection['test_evaluated'] and selection['selected']=='PMM_C'
seeds=[1,2,3,4,5]
run_ids=['GMM_fixed_formal',selection['run_ids'][selection['selected']]]
write_json(OUT/'sampling_sensitivity_plan.json',{
    'inference_seeds':seeds,'run_ids':run_ids,'trained_weights_fixed':True,
    'no_retraining':True,'no_seed_selection':True,'primary_results_unchanged':True,
    'purpose':'Assess 1000-sample forecast Monte Carlo variation because the primary test scores differ by only about 0.0006.',
    'selected_at_epoch':selection['selected_at_epoch'],'started_epoch':time.time(),
    'limitation':'Inference sampling variability only; not uncertainty across training seeds or datasets.'})
data=load_data()
y,s,tags,train,valid,pretest,test=data
qs=np.concatenate([np.arange(1,50),np.arange(51,100)])/100
cols=[f'NHITS-lo-{lv}' for lv in reversed(LEVELS)]+[f'NHITS-hi-{lv}' for lv in LEVELS]
rows=[]
for seed in seeds:
    for run in run_ids:
        if seed==1:
            pred=pd.read_parquet(OUT/f'{run}_test_predictions.parquet')
        else:
            pred=predict_saved_model(run,pretest,s,sampling_seed=seed)
        pred=pred.rename(columns=lambda c:c.replace('.0',''))
        joined=test[['unique_id','ds','y']].merge(pred,on=['unique_id','ds'],validate='one_to_one')
        error=joined.y.to_numpy()[:,None]-joined[cols].to_numpy()
        value=float(2*np.maximum(error*qs,error*(qs-1)).sum()/(len(qs)*joined.y.abs().sum()))
        rows.append({'run_id':run,'inference_seed':seed,'pooled_scrps':value})
        print('SAMPLING',run,seed,value,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'sampling_sensitivity.csv',index=False)
frame=pd.DataFrame(rows)
summary=frame.groupby('run_id').pooled_scrps.agg(['mean','std','min','max'])
summary.to_csv(OUT/'sampling_sensitivity_summary.csv')
wide=frame.pivot(index='inference_seed',columns='run_id',values='pooled_scrps')
diff=wide[run_ids[1]]-wide[run_ids[0]]
write_json(OUT/'sampling_sensitivity_summary.json',{'seeds':seeds,
    'summary':summary.to_dict(orient='index'),
    'PMM_minus_GMM_difference_mean':float(diff.mean()),
    'PMM_minus_GMM_difference_std':float(diff.std()),
    'PMM_lower_score_inference_seed_count':int((diff<0).sum()),
    'not_training_seed_variation':True,'not_a_statistical_significance_test':True})
