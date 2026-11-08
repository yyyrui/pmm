"""Inspect saved PMM rates on validation inputs only; does not retrain."""
from experiment_code import *
import sys

name = sys.argv[1] if len(sys.argv) > 1 else 'PMM_A_short'
record = json.loads((OUT/f'{name}.json').read_text())
data = load_data()
train = data[3]
loss = PMM(n_components=10,quantiles=QUANTILES,weighted=record.get('weighted',False))
model = NHITS(h=12,input_size=24,loss=loss,hist_exog_list=['month'],
              max_steps=record['max_steps'],scaler_type='robust',learning_rate=record['lr'],
              valid_loss=sCRPS(quantiles=QUANTILES),random_seed=1,
              accelerator='cpu',devices=1,logger=False,enable_progress_bar=False,
              enable_model_summary=False)
model.load_state_dict(torch.load(OUT/f'{name}_weights.pt',weights_only=True))
collected=[]
original=loss.scale_decouple
def capture(*args,**kwargs):
    result=original(*args,**kwargs)
    rates=result[0].detach().cpu().numpy().reshape(-1,10)
    weights=result[1].detach().cpu().numpy().reshape(-1,10) if len(result)>1 else np.full_like(rates,.1)
    collected.append((rates,weights))
    return result
loss.scale_decouple=capture
dataset=TimeSeriesDataset.from_df(train)[0]
model.predict(dataset,random_seed=1)
rates=np.concatenate([a for a,b in collected])
weights=np.concatenate([b for a,b in collected])
np.savez_compressed(OUT/f'{name}_validation_distribution.npz',rates=rates,weights=weights)
mean=(rates*weights).sum(axis=-1)
between=((rates-mean[:,None])**2*weights).sum(axis=-1)
result={'name':name,'phase':'validation_inputs_only','forecast_points':len(rates),
        'rate_min':float(rates.min()),'rate_max':float(rates.max()),
        'component_rate_floor_fraction':float((rates<=.00101).mean()),
        'all_components_at_floor_fraction':float((rates<=.00101).all(axis=-1).mean()),
        'weighted_rate_floor_mass':float((weights*(rates<=.00101)).sum(axis=-1).mean()),
        'mean_weight_entropy':float(-(weights*np.log(np.clip(weights,1e-12,1))).sum(axis=-1).mean()),
        'median_mixture_variance_over_mean':float(np.median((mean+between)/np.maximum(mean,1e-9))),
        'note':'Rates at the softplus floor may have near-zero gradients; this inference probe does not measure training gradients or prove causation.'}
# Posterior assignment diagnostics use validation targets only. Compare PMM
# components within the same model; do not compare discrete and continuous NLLs.
target=data[4]['y'].to_numpy()
assert len(target)==len(rates)
from scipy.special import gammaln, logsumexp
loglik=target[:,None]*np.log(rates)-rates-gammaln(target[:,None]+1)
joint=loglik+np.log(np.clip(weights,1e-35,1))
posterior=np.exp(joint-logsumexp(joint,axis=1,keepdims=True))
result.update({
    'validation_mean_NLL_with_fractional_extension':float(-logsumexp(joint,axis=1).mean()),
    'prior_component_weight_mean':weights.mean(axis=0).tolist(),
    'posterior_component_responsibility_mean':posterior.mean(axis=0).tolist(),
    'mean_effective_prior_components':float(np.exp(-(weights*np.log(np.clip(weights,1e-35,1))).sum(axis=1)).mean()),
    'mean_effective_posterior_components':float(np.exp(-(posterior*np.log(np.clip(posterior,1e-35,1))).sum(axis=1)).mean()),
    'posterior_below_1e_minus_6_fraction':float((posterior<1e-6).mean()),
    'raw_mean_bias':float((mean-target).mean()),
    'bottom_raw_mean_bias':float((mean.reshape(len(data[1]),12)[-data[1].shape[1]:]-target.reshape(len(data[1]),12)[-data[1].shape[1]:]).mean()),
    'diagnostic_caution':'Posterior responsibility concentration is conditional on each observation; by itself it does not prove a component is unused or cause forecast error.'})
write_json(OUT/f'{name}_rate_diagnostics.json',result)
print(json.dumps(result,indent=2),flush=True)
