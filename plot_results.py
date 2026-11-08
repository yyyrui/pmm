"""Export validation comparison and calibration charts from saved predictions."""
from experiment_code import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y,s,tags,train,valid,pretest,test=load_data()
run_ids=['GMM_fixed_formal','PMM_A_formal','PMM_B_extension','PMM_C_extension']
colors=['#2563eb','#dc2626','#f59e0b','#059669']
labels=['GMM (aligned)','PMM A: equal, 1e-3','PMM B: equal, 3e-4','PMM C: learned, 3e-4']
records=[json.loads((OUT/f'{run}.json').read_text()) for run in run_ids]
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
values=[r['validation_scrps'] for r in records]
bars=axes[0].bar(range(4),values,color=colors)
axes[0].set_xticks(range(4),labels,rotation=18,ha='right',fontsize=9)
axes[0].bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3)
axes[0].set_ylim(0,max(values)*1.2)
axes[0].set_ylabel('Validation pooled sCRPS (lower is better)')
axes[0].set_title('Same 2015 validation period; 3 PMM settings')
truth=valid[valid.unique_id=='TotalAll'].sort_values('ds')
axes[1].plot(truth.ds,truth.y,color='black',marker='o',label='Observed',linewidth=2)
for run,label,color in zip(run_ids,labels,colors):
    pred=pd.read_parquet(OUT/f'{run}_validation_predictions.parquet')
    part=pred[pred.unique_id=='TotalAll'].sort_values('ds')
    axes[1].plot(part.ds,part['NHITS'],label=label,color=color)
axes[1].set_title('Country level: validation forecast means')
axes[1].set_ylabel('TourismLarge target (original units)')
axes[1].tick_params(axis='x',rotation=30)
axes[1].legend(fontsize=8)
fig.savefig(OUT/'validation_comparison.png',dpi=180)
plt.close(fig)

# Test chart is descriptive only, after the selected setting was frozen.
selection=json.loads((OUT/'selection.json').read_text())
selected_id=selection['run_ids'][selection['selected']]
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
truth=test[test.unique_id=='TotalAll'].sort_values('ds')
for ax,run,label,color in zip(axes,['GMM_fixed_formal',selected_id],['GMM','Selected PMM'],['#2563eb','#059669']):
    p=pd.read_parquet(OUT/f'{run}_test_predictions.parquet').rename(columns=lambda x:x.replace('.0',''))
    p=p[p.unique_id=='TotalAll'].sort_values('ds')
    ax.fill_between(p.ds,p['NHITS-lo-90'],p['NHITS-hi-90'],alpha=.2,color=color,label='90% interval')
    ax.plot(p.ds,p['NHITS'],color=color,label='Forecast mean')
    ax.plot(truth.ds,truth.y,color='black',marker='o',label='Observed')
    ax.set_title(label+' — 2016 test, country level')
    ax.set_ylabel('Original target units')
    ax.legend(fontsize=8)
axes[-1].tick_params(axis='x',rotation=30)
fig.suptitle('Extension reuses a previously viewed test period; descriptive comparison')
fig.savefig(OUT/'test_country_intervals.png',dpi=180)
plt.close(fig)
print('Saved validation_comparison.png and test_country_intervals.png')
