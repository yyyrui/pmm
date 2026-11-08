"""Reproducible local PMM checks; never overwrite the October experiments.

Run with the project's .venv. External interpreter use requires the user's
approval because this project's .venv/bin/python is a symbolic link.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'pmm_checks_20261108'
RUNTIME = OUT / 'runtime'
for key, folder in {
    'TMPDIR': 'tmp', 'MPLCONFIGDIR': 'matplotlib', 'XDG_CACHE_HOME': 'cache',
    'TORCH_HOME': 'torch', 'IPYTHONDIR': 'ipython', 'JUPYTER_CONFIG_DIR': 'jupyter',
}.items():
    location = RUNTIME / folder
    location.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(location)
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, HINT
from neuralforecast.losses.pytorch import GMM, PMM, sCRPS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import augment_calendar_df
from utilsforecast.losses import scaled_crps

torch.set_num_threads(1)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
H = 12
LEVELS = list(range(2, 100, 2))
QUANTILES = np.arange(1, 100) / 100
EVAL_Q = np.r_[np.arange(1, 50), np.arange(51, 100)] / 100
QCOLS = ([f'NHITS-lo-{v}' for v in reversed(LEVELS)] +
         [f'NHITS-hi-{v}' for v in LEVELS])
CONFIGS = {
    'GMM': dict(loss='GMM', lr=0.001, weighted=False),
    'PMM_fixed': dict(loss='PMM', lr=0.0003, weighted=False),
    'PMM_weighted': dict(loss='PMM', lr=0.0003, weighted=True),
}
SEEDS = [1, 7, 21]
YEARS = [2014, 2015]
STEPS = 2000
TAG_NAMES = ['Country', 'Country/State', 'Country/State/Zone',
             'Country/State/Zone/Region', 'Country/Purpose',
             'Country/State/Purpose', 'Country/State/Zone/Purpose',
             'Country/State/Zone/Region/Purpose']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False, default=str))
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_local_data():
    """Read only the local CSVs, with no download/cache fallback."""
    folder = ROOT / 'data/hierarchical/TourismLarge'
    s = pd.read_csv(folder / 'agg_mat.csv', index_col=0)
    wide = pd.read_csv(folder / 'data.csv', index_col=0).T
    y = wide.stack().rename('y').rename_axis(['unique_id', 'ds']).reset_index()
    y['ds'] = pd.to_datetime(y['ds'])
    y, _ = augment_calendar_df(y, freq='MS')
    raw_ids = TimeSeriesDataset.from_df(y)[1]
    dtype = pd.CategoricalDtype(categories=s.index.tolist(), ordered=True)
    y['unique_id'] = y['unique_id'].astype(dtype)
    y = y.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    ids = TimeSeriesDataset.from_df(y)[1]
    assert list(ids) == list(s.index)
    assert list(s.index[-s.shape[1]:]) == list(s.columns)
    assert np.array_equal(s.iloc[-s.shape[1]:].to_numpy(), np.eye(s.shape[1]))
    assert y['unique_id'].notna().all() and not y.duplicated(['unique_id', 'ds']).any()
    assert np.isfinite(y.y).all() and (y.y >= 0).all()
    assert y.groupby('unique_id', observed=True).size().nunique() == 1
    cuts, = np.where(s.sum(axis=1).cumsum() % s.shape[1] == 0)
    groups = [s.index[:1].to_numpy()] + [
        s.index[cuts[i] + 1:cuts[i + 1] + 1].to_numpy()
        for i in range(len(cuts) - 1)]
    assert len(groups) == len(TAG_NAMES) and sum(map(len, groups)) == len(s)
    tags = dict(zip(TAG_NAMES, groups))
    # Validate observed hierarchy on pre-2016 data only.
    observed = y[y.ds.dt.year < 2016].pivot(index='unique_id', columns='ds', values='y')
    observed = observed.reindex(s.index)
    mismatch = np.abs(observed.to_numpy() - s.to_numpy() @ observed.reindex(s.columns).to_numpy()).max()
    assert mismatch < 1e-6
    audit = dict(series=len(s), rows=len(y), S_shape=list(s.shape),
                 raw_order_matches_S=list(raw_ids) == list(s.index),
                 corrected_order_matches_S=True, bottom_identity=True,
                 fractional_fraction=float((abs(y.y - y.y.round()) > 1e-6).mean()),
                 observed_coherence_max_error=float(mismatch),
                 data_sha256=sha256(folder / 'data.csv'),
                 S_sha256=sha256(folder / 'agg_mat.csv'))
    return y, s, tags, audit


def split_year(y, year):
    history = y[y.ds.dt.year < year].copy()
    valid = y[y.ds.dt.year == year].copy()
    fit_frame = y[y.ds.dt.year <= year].copy()
    assert history.ds.max() < valid.ds.min()
    assert valid.groupby('unique_id', observed=True).size().eq(H).all()
    assert fit_frame.ds.max() < pd.Timestamp('2016-01-01')
    return history, valid, fit_frame


def score(pred, truth, s, tags):
    pred = pred.rename(columns=lambda c: c.replace('.0', ''))
    joined = truth[['unique_id', 'ds', 'y']].merge(pred, on=['unique_id', 'ds'],
                                                validate='one_to_one')
    assert len(pred) == len(truth) == len(joined)
    all_qcols = QCOLS[:49] + ['NHITS-median'] + QCOLS[49:]
    assert np.isfinite(joined[['NHITS'] + all_qcols].to_numpy()).all()
    assert (np.diff(joined[all_qcols].to_numpy(), axis=1) >= -1e-8).all()
    rows = []
    for level, ids in {**tags, 'Overall': s.index.to_numpy()}.items():
        part = joined[joined.unique_id.isin(ids)]
        error = part.y.to_numpy()[:, None] - part[QCOLS].to_numpy()
        denom = float(part.y.abs().sum())
        pooled = float(2 * np.maximum(EVAL_Q * error, (EVAL_Q - 1) * error).sum() /
                       (len(EVAL_Q) * denom)) if denom > 0 else None
        residual = part.NHITS - part.y
        rows.append(dict(level=level, pooled_scrps=pooled,
                         mae=float(residual.abs().mean()),
                         rmse=float(np.sqrt(np.mean(residual ** 2))),
                         wape=float(residual.abs().sum() / denom) if denom else None,
                         bias=float(residual.mean()),
                         coverage90=float(((part.y >= part['NHITS-lo-90']) &
                                           (part.y <= part['NHITS-hi-90'])).mean()),
                         width90=float((part['NHITS-hi-90'] - part['NHITS-lo-90']).mean()),
                         negative_mean_count=int((part.NHITS < 0).sum()),
                         zero_total_series=int((part.groupby('unique_id', observed=True).y.sum() == 0).sum())))
    metrics = pd.DataFrame(rows)
    pivot = pred.pivot(index='unique_id', columns='ds', values='NHITS').reindex(s.index)
    coherent = s.to_numpy() @ pivot.reindex(s.columns).to_numpy()
    np.testing.assert_allclose(pivot.to_numpy(), coherent, rtol=1e-6, atol=1e-5)
    reference = joined[['unique_id', 'y'] + QCOLS].copy()
    per_series = scaled_crps(reference, models={'NHITS': QCOLS}, quantiles=EVAL_Q)
    metrics['native_series_scaled_crps_mean'] = [
        float(per_series.loc[per_series.unique_id.isin(ids), 'NHITS'].mean())
        for ids in [*tags.values(), s.index.to_numpy()]]
    reference['unique_id'] = 'pooled'
    ref = float(scaled_crps(reference, models={'NHITS': QCOLS}, quantiles=EVAL_Q).NHITS.iloc[0])
    overall = rows[-1]
    assert np.isclose(overall['pooled_scrps'], ref, rtol=1e-10, atol=1e-12)
    checks = dict(rows=len(pred), finite=True, quantiles_monotone=True,
                  mean_coherence_max_error=float(abs(pivot.to_numpy() - coherent).max()),
                  metric_matches_library_after_pooling=True)
    return metrics, checks


class CaptureWeights(Callback):
    """Capture native-validation best weights without changing optimization."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.best = float('inf')
        self.best_step = None
        self.started = time.time()
        self.observations = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        value = float(trainer.callback_metrics['ptl/val_loss'])
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite native validation loss')
        if value < self.best:
            self.best, self.best_step = value, int(trainer.global_step)
            torch.save(pl_module.state_dict(), self.directory / 'best_weights.pt')
        row = dict(step=int(trainer.global_step), native_valid_loss=value,
                   best_step=self.best_step, best_native_valid_loss=self.best,
                   elapsed_seconds=round(time.time() - self.started, 2))
        self.observations.append(row)
        write_json(self.directory / 'progress.json', row)
        if trainer.global_step % 250 == 0:
            print('PROGRESS', self.directory.name, json.dumps(row), flush=True)


def make_base(config, seed, steps=STEPS, callbacks=None):
    loss_class = GMM if config['loss'] == 'GMM' else PMM
    return NHITS(h=H, input_size=24,
                 loss=loss_class(n_components=10, quantiles=QUANTILES,
                                 weighted=config['weighted'], num_samples=1000),
                 hist_exog_list=['month'], max_steps=steps,
                 early_stop_patience_steps=10, val_check_steps=min(50, steps),
                 scaler_type='robust', learning_rate=config['lr'],
                 valid_loss=sCRPS(quantiles=QUANTILES), random_seed=seed,
                 accelerator='cpu', devices=1, logger=False,
                 enable_progress_bar=False, enable_model_summary=False,
                 default_root_dir=str(OUT), callbacks=list(callbacks or []))


def forecast(base, history, s, sampling_seed=1):
    dataset, uids, last_dates, *_ = TimeSeriesDataset.from_df(history)
    assert list(uids) == list(s.index)
    hint = HINT(h=H, model=base, S=s.to_numpy(), reconciliation='BottomUp', alias='NHITS')
    np.random.seed(sampling_seed)
    values = hint.predict(dataset, random_seed=sampling_seed)
    dates = np.concatenate([pd.date_range(pd.Timestamp(date) + pd.offsets.MonthBegin(1),
                                         periods=H, freq='MS').to_numpy()
                            for date in last_dates])
    pred = pd.DataFrame(values, columns=['NHITS' + c for c in base.loss.output_names])
    pred.insert(0, 'ds', dates)
    pred.insert(0, 'unique_id', np.repeat(np.asarray(uids), H))
    return pred.rename(columns=lambda c: c.replace('.0', ''))


def fit_one(name, year, seed, steps=STEPS, smoke=False):
    run_id = f'{name}_y{year}_s{seed}'
    directory = OUT / ('smoke' if smoke else 'runs') / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('status') == 'completed' and old['max_steps'] == steps:
            print('ALREADY_COMPLETED', run_id, flush=True)
            return old
    started = time.time()
    record = dict(run_id=run_id, name=name, validation_year=year, training_seed=seed,
                  sampling_seed=1, config=CONFIGS[name], max_steps=steps,
                  status='running', source_sha256=sha256(__file__),
                  test_evaluated=False)
    write_json(path, record)
    try:
        y, s, tags, audit = load_local_data()
        history, valid, fit_frame = split_year(y, year)
        callback = CaptureWeights(directory)
        base = make_base(CONFIGS[name], seed, steps, [callback])
        nf = NeuralForecast(models=[HINT(h=H, model=base, S=s.to_numpy(),
                                        reconciliation='BottomUp', alias='NHITS')], freq='MS')
        nf.fit(df=fit_frame, val_size=H)
        assert list(nf.uids) == list(s.index)
        base = nf.models[0].model
        # NeuralForecast deep-copies the supplied model and its callbacks.
        # Read the callback belonging to the trained copy, not the initializer.
        callback = next(cb for cb in base.trainer_kwargs['callbacks']
                        if isinstance(cb, CaptureWeights))
        final_step = len(base.train_trajectories)
        assert callback.best_step is not None and callback.best_step <= final_step
        torch.save(base.state_dict(), directory / 'last_weights.pt')
        record.update(actual_steps=final_step, best_step=callback.best_step,
                      best_native_valid_loss=callback.best,
                      final_native_valid_loss=callback.observations[-1]['native_valid_loss'],
                      stop_reason='max_steps' if final_step >= steps else 'early_stopping',
                      gradient_train_end=str(history.ds.max().date()),
                      validation_start=str(valid.ds.min().date()),
                      validation_end=str(valid.ds.max().date()), data_audit=audit)
        write_json(directory / 'trajectories.json', dict(
            training=base.train_trajectories, validation=callback.observations))
        # on_validation_end never runs during predict; remove callbacks to make
        # saved-model inference independent of the training callback instance.
        base.trainer_kwargs['callbacks'] = []
        for policy in ['last', 'best']:
            if policy == 'best':
                base.load_state_dict(torch.load(directory / 'best_weights.pt', weights_only=True))
            pred = forecast(base, history, s)
            metrics, checks = score(pred, valid, s, tags)
            pred.to_parquet(directory / f'{policy}_validation_predictions.parquet', index=False)
            metrics.to_csv(directory / f'{policy}_validation_metrics.csv', index=False)
            row = metrics.loc[metrics.level == 'Overall'].iloc[0].to_dict()
            record[policy] = dict(metrics=row, checks=checks)
        # Check that the saved model can reproduce predictions exactly.
        restored = make_base(CONFIGS[name], seed, steps)
        restored.trainer_kwargs['callbacks'] = []
        restored.load_state_dict(torch.load(directory / 'best_weights.pt', weights_only=True))
        reloaded = forecast(restored, history, s)
        saved = pd.read_parquet(directory / 'best_validation_predictions.parquet')
        cols = [c for c in saved if c not in ['unique_id', 'ds']]
        np.testing.assert_array_equal(reloaded[cols].to_numpy(), saved[cols].to_numpy())
        record['reload_exact_match'] = True
        record['status'] = 'completed'
        record['elapsed_seconds'] = round(time.time() - started, 2)
        write_json(path, record)
        print('COMPLETED', json.dumps(record, ensure_ascii=False), flush=True)
        return record
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), elapsed_seconds=time.time() - started)
        write_json(path, record)
        raise


def initialize_plan():
    plan = dict(date='2026-11-08', validation_years=YEARS, training_seeds=SEEDS,
                configs=CONFIGS, max_steps=STEPS, runs=18,
                common=dict(horizon=H, input_size=24, components=10, samples=1000,
                            scaler='robust', val_check_steps=50, early_stop_patience=10,
                            sampling_seed=1),
                checkpoints=['last', 'best'],
                best_definition='lowest native ptl/val_loss, excluding sanity check',
                selection='minimum mean validation Overall pooled sCRPS across both years and all three training seeds; select PMM configuration/checkpoint and GMM checkpoint separately',
                test_rule='2016 scored only after complete validation selection, for the selected PMM and GMM policies, all 3 prespecified seeds; reused period is exploratory, not independent confirmation',
                limitations=['Rolling years share training history; six runs are not six independent datasets.',
                             '2015 and 2016 have been inspected in the existing October experiments.',
                             'Native validation is before reconciliation; final comparison is after BottomUp.',
                             'No rounding or rescaling of target units.'],
                preserved_files={str(p.relative_to(ROOT)): sha256(p) for p in [
                    ROOT / 'hierarchical_forecasting.ipynb', ROOT / 'ＰＰＰ.ipynb',
                    ROOT / 'experiment_results/comparison.csv', ROOT / 'experiment_results/selection.json']})
    path = OUT / 'plan.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous == plan, 'Plan or preserved files changed; inspect before proceeding'
    else:
        write_json(path, plan)
    environment = {p: importlib.metadata.version(p) for p in [
        'neuralforecast', 'torch', 'pytorch-lightning', 'pandas', 'numpy',
        'utilsforecast', 'datasetsforecast', 'hierarchicalforecast']}
    environment.update(python=sys.version, executable=sys.executable,
                       resolved_executable=str(Path(sys.executable).resolve()),
                       runtime_environment={k: os.environ[k] for k in [
                           'TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME',
                           'IPYTHONDIR', 'JUPYTER_CONFIG_DIR']})
    write_json(OUT / 'environment.json', environment)
    return plan


def audit_data():
    plan = initialize_plan()
    y, s, tags, audit = load_local_data()
    folds = []
    for year in YEARS:
        history, valid, fit_frame = split_year(y, year)
        # A seasonal naive point forecast, represented as a degenerate distribution.
        naive = history.groupby('unique_id', observed=True).tail(H)[['unique_id', 'ds', 'y']].copy()
        naive['ds'] = naive.ds + pd.DateOffset(years=1)
        naive = pd.concat([naive, pd.DataFrame(
            np.repeat(naive.y.to_numpy()[:, None], 100, axis=1),
            columns=['NHITS', 'NHITS-median'] + QCOLS, index=naive.index)], axis=1)
        metrics, checks = score(naive.drop(columns='y'), valid, s, tags)
        metrics.to_csv(OUT / f'seasonal_naive_{year}.csv', index=False)
        folds.append(dict(year=year, train_start=str(history.ds.min().date()),
                          train_end=str(history.ds.max().date()),
                          validation_start=str(valid.ds.min().date()),
                          validation_end=str(valid.ds.max().date()),
                          train_rows=len(history), validation_rows=len(valid),
                          zero_total_validation_series=int((valid.groupby('unique_id', observed=True).y.sum() == 0).sum()),
                          seasonal_naive_overall=metrics.iloc[-1].to_dict(), checks=checks))
    audit['folds'] = folds
    audit['preserved_files_match'] = all(sha256(ROOT / p) == h for p, h in plan['preserved_files'].items())
    write_json(OUT / 'data_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def run_all(workers=2):
    initialize_plan()
    jobs = [(name, year, seed) for year in reversed(YEARS) for seed in SEEDS for name in CONFIGS]
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    def launch(job):
        name, year, seed = job
        run_id = f'{name}_y{year}_s{seed}'
        with (logs / f'{run_id}.log').open('a') as log:
            process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                      'run', '--name', name, '--year', str(year),
                                      '--seed', str(seed)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        return run_id, process.returncode
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            run_id, code = future.result()
            print('WORKER_FINISHED', run_id, 'exit', code, flush=True)
            if code:
                failures.append(run_id)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')
    summarize()


def summarize():
    rows = []
    for name in CONFIGS:
        for year in YEARS:
            for seed in SEEDS:
                path = OUT / 'runs' / f'{name}_y{year}_s{seed}' / 'result.json'
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                if r['status'] != 'completed':
                    continue
                for policy in ['last', 'best']:
                    rows.append(dict(name=name, year=year, seed=seed, policy=policy,
                                     actual_steps=r['actual_steps'], best_step=r['best_step'],
                                     **r[policy]['metrics']))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(OUT / 'validation_runs.csv', index=False)
    aggregate = df.groupby(['name', 'policy']).agg(
        runs=('pooled_scrps', 'count'), mean_scrps=('pooled_scrps', 'mean'),
        std_scrps=('pooled_scrps', 'std'), min_scrps=('pooled_scrps', 'min'),
        max_scrps=('pooled_scrps', 'max'), mean_mae=('mae', 'mean'),
        mean_coverage90=('coverage90', 'mean')).reset_index()
    aggregate.to_csv(OUT / 'validation_summary.csv', index=False)
    print(aggregate.to_string(index=False), flush=True)
    if len(rows) != 36:
        return df
    pmm = aggregate[aggregate.name.str.startswith('PMM')].sort_values(['mean_scrps', 'name', 'policy']).iloc[0]
    gmm = aggregate[aggregate.name == 'GMM'].sort_values(['mean_scrps', 'policy']).iloc[0]
    selected = dict(selected_PMM={'name': pmm['name'], 'policy': pmm['policy'],
                                 'mean_validation_scrps': float(pmm.mean_scrps)},
                    GMM_baseline={'name': gmm['name'], 'policy': gmm['policy'],
                                  'mean_validation_scrps': float(gmm.mean_scrps)},
                    complete_training_runs=18, selected_before_new_test=True,
                    test_evaluated=False, representative_seed=1,
                    test_period_previously_seen=True)
    selection_path = OUT / 'selection.json'
    if selection_path.exists():
        frozen = json.loads(selection_path.read_text())
        assert all(frozen[k] == selected[k] for k in ['selected_PMM', 'GMM_baseline']), \
            'Frozen selection differs; do not silently replace it'
        selected['test_evaluated'] = frozen['test_evaluated']
    write_json(selection_path, selected)
    return df


def evaluate_selected_test():
    # Read the validation-only choice frozen before accessing test targets.
    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text())
    assert selection['complete_training_runs'] == 18
    y, s, tags, _ = load_local_data()
    history = y[y.ds.dt.year < 2016].copy()
    truth = y[y.ds.dt.year == 2016].copy()
    directory = OUT / 'test'
    directory.mkdir(exist_ok=True)
    rows = []
    for chosen in [selection['selected_PMM'], selection['GMM_baseline']]:
        name, policy = chosen['name'], chosen['policy']
        for seed in SEEDS:
            model_dir = OUT / 'runs' / f'{name}_y2015_s{seed}'
            base = make_base(CONFIGS[name], seed)
            base.trainer_kwargs['callbacks'] = []
            base.load_state_dict(torch.load(model_dir / f'{policy}_weights.pt', weights_only=True))
            pred = forecast(base, history, s)
            metrics, checks = score(pred, truth, s, tags)
            stem = f'{name}_{policy}_s{seed}'
            pred.to_parquet(directory / f'{stem}_predictions.parquet', index=False)
            metrics.to_csv(directory / f'{stem}_metrics.csv', index=False)
            write_json(directory / f'{stem}_checks.json', checks)
            rows.append(dict(name=name, policy=policy, seed=seed, **metrics.iloc[-1].to_dict()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'test_runs.csv', index=False)
    selection['test_evaluated'] = True
    write_json(selection_path, selection)
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'run', 'all', 'summarize', 'test'])
    parser.add_argument('--name', choices=list(CONFIGS), default='PMM_weighted')
    parser.add_argument('--year', type=int, choices=YEARS, default=2015)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.action == 'audit':
        audit_data()
    elif args.action == 'run':
        if not args.smoke and (args.steps != STEPS or args.seed not in SEEDS):
            parser.error('Formal runs must use the prespecified steps and seeds')
        fit_one(args.name, args.year, args.seed, args.steps, args.smoke)
    elif args.action == 'all':
        run_all(args.workers)
    elif args.action == 'summarize':
        summarize()
    else:
        evaluate_selected_test()
