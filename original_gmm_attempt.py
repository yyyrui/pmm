"""Execute original GMM cells verbatim without modifying the source notebook.

The ARIMA comparison/plots are outside this GMM vs PMM experiment. Test forecasts
are saved without evaluating test targets; selection must finish first.
"""
import contextlib
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.mpl-cache').resolve()))
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path('.numba-cache').resolve()))
out = Path('experiment_results')
original = Path('hierarchical_forecasting.ipynb')
notebook = json.loads(original.read_text())
namespace = {}
started = time.time()
record = {'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
          'cells': [], 'test_evaluated': False}
with (out / 'original_gmm_attempt.log').open('w') as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            for idx in [7, 8, 13, 14, 15, 16, 17]:
                source = ''.join(notebook['cells'][idx]['source'])
                # IPython capture only suppresses display; pure Python omits it.
                source = source.replace('%%capture\n', '')
                print(f'EXECUTING ORIGINAL CELL {idx}', flush=True)
                record['cells'].append(idx)
                exec(compile(source, f'original_cell_{idx}', 'exec'), namespace)
                log.flush()
            namespace['Y_hat_df'].to_parquet(out / 'original_gmm_test_predictions.parquet')
            nf = namespace['nf']
            trained = nf.models[0].model
            record['train_trajectories'] = trained.train_trajectories
            record['valid_trajectories'] = trained.valid_trajectories
            record['actual_steps'] = len(trained.train_trajectories)
            record['model_name'] = str(nf.models[0])
            record['uids_match_S_order'] = list(nf.uids) == namespace['S_df']['unique_id'].tolist()
            # Validate the original trained model without using test targets.
            train = namespace['Y_train_df']
            val = train.groupby('unique_id', observed=True).tail(12)
            history = train.drop(val.index)
            pred_val = nf.predict(df=history)
            pred_val.to_parquet(out / 'original_gmm_validation_predictions.parquet')
            import torch
            torch.save(trained.state_dict(), out / 'original_gmm_weights.pt')
            record['status'] = 'completed'
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
            traceback.print_exc()
        finally:
            record['elapsed_seconds'] = time.time() - started
            (out / 'original_gmm_attempt.json').write_text(json.dumps(record, indent=2, default=float))
print(json.dumps({k:v for k,v in record.items() if not k.endswith('trajectories')}, indent=2, default=float))
