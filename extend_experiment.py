"""Complete the two predeclared PMM settings; preserve the first-round results."""
from experiment_code import *
from datetime import datetime

deadline=datetime.fromisoformat('2026-10-29T14:03:57+08:00').timestamp()
data=load_data()
plan={'phase':'extension','settings_predeclared_in_first_round':True,
      'unique_PMM_settings_total':3,'configs':[
          {'name':'PMM_B','loss':'PMM','lr':3e-4},
          {'name':'PMM_C','loss':'PMM','lr':3e-4,'weighted':True}],
      'selection_metric':'validation Overall pooled_scaled_crps',
      'previous_test_results_already_seen':True,
      'new_test_results_exploratory':True,
      'deadline_epoch':deadline,'started_epoch':time.time(),
      'restart_B_from_seed_1':True,'restart_reason':'First-round run had no saved weights or optimizer state.'}
write_json(OUT/'extension_plan.json',plan)
runs={}
for cfg in plan['configs']:
    nf,record=fit_run(cfg,2000,data,deadline-120,phase='extension')
    if nf is not None and record.get('status')=='completed':
        runs[cfg['name']]=(nf,record)

a=json.loads((OUT/'PMM_A_formal.json').read_text())
candidates={'PMM_A':a,**{name:rec for name,(nf,rec) in runs.items()}}
selected=min(candidates,key=lambda name:candidates[name]['validation_scrps'])
selection={'selected':selected,'phase':'extension','max_steps':2000,
           'selection_metric':plan['selection_metric'],
           'validation_scores':{name:rec['validation_scrps'] for name,rec in candidates.items()},
           'run_ids':{name:name+('_formal' if name=='PMM_A' else '_extension') for name in candidates},
           'excluded':{cfg['name']:'incomplete' for cfg in plan['configs'] if cfg['name'] not in runs},
           'selected_at_epoch':time.time(),'test_evaluated':False,
           'previous_test_results_already_seen':True,
           'limitation':'Extension of predeclared validation search; test period is reused and is not a fresh independent confirmation.'}
write_json(OUT/'selection_extension.json',selection)
# Test access is after validation-only selection. Do not score unselected B/C.
if selected=='PMM_A':
    selection['test_scrps']=a['test_scrps']
    selection['test_reused_without_reevaluation']=True
else:
    nf,rec=runs[selected]
    y,s,tags,train,valid,pretest,test=data
    np.random.seed(SEED)
    pred=nf.predict(df=pretest,random_seed=SEED)
    run_id=selected+'_extension'
    pred.to_parquet(OUT/f'{run_id}_test_predictions.parquet')
    rec['test_scrps']=score(pred,test,tags,pretest,run_id,'test')
    rec['test_evaluated']=True
    rec['test_is_extension_exploratory']=True
    write_json(OUT/f'{run_id}.json',rec)
    selection['test_scrps']=rec['test_scrps']
selection['test_evaluated']=True
selection['finished_epoch']=time.time()
write_json(OUT/'selection_extension.json',selection)
write_json(OUT/'selection.json',selection)
write_json(OUT/'extension_done.json',{'status':'completed','finished_epoch':time.time(),
                                  'completed_settings':list(candidates),'selected':selected})
print('EXTENSION_COMPLETE',json.dumps(selection),flush=True)
