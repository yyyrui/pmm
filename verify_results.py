"""Verify preservation, splits, and pooled score against the library implementation."""
from experiment_code import *

session=json.loads((OUT/'session.json').read_text())
assert hashlib.sha256(Path(session['original_file']).read_bytes()).hexdigest()==session['original_sha256']
y,s,tags,train,valid,pretest,test=load_data()
assert train.ds.max()<valid.ds.min() and valid.ds.max()<test.ds.min()
assert train.ds.nunique()==204 and valid.ds.nunique()==12 and test.ds.nunique()==12
cols=[f'NHITS-lo-{lv}' for lv in reversed(LEVELS)]+[f'NHITS-hi-{lv}' for lv in LEVELS]
qs=np.concatenate([np.arange(1,50),np.arange(51,100)])/100
checks=[]
for path in sorted(OUT.glob('*_validation_pooled_metrics.csv')):
    stem=path.name.replace('_validation_pooled_metrics.csv','')
    pred_path=OUT/f'{stem}_validation_predictions.parquet'
    if not pred_path.exists():
        continue
    pred=pd.read_parquet(pred_path).rename(columns=lambda c:c.replace('HINT','NHITS').replace('.0',''))
    joined=valid[['unique_id','ds','y']].merge(pred,on=['unique_id','ds'],validate='one_to_one')
    assert len(joined)==6660 and np.isfinite(joined[cols]).all().all()
    joined['unique_id']='pooled'
    ref=scaled_crps(joined,models={'NHITS':cols},quantiles=qs)['NHITS'].iloc[0]
    actual=pd.read_csv(path).query('level=="Overall"')['NHITS'].iloc[0]
    assert np.isclose(ref,actual,rtol=1e-10,atol=1e-12),(stem,ref,actual)
    checks.append({'run':stem,'manual_pooled':float(actual),'library_pooled':float(ref)})
write_json(OUT/'verification.json',{'original_unchanged':True,'split_disjoint':True,
    'pooled_formula_matches_library_after_pooling_ids':True,'checks':checks})
print('VERIFIED',len(checks),'validation result sets; original unchanged.',flush=True)
