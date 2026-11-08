"""Build the follow-up report and a small, executable improved PMM notebook."""
from pmm_robustness import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md_table(frame, digits=6):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}'
        return str(value)
    headers = list(frame.columns)
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(fmt(v) for v in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def verification():
    plan = json.loads((OUT / 'plan.json').read_text())
    preserved = {p: sha256(ROOT / p) == h for p, h in plan['preserved_files'].items()}
    assert all(preserved.values()), preserved
    checks = []
    source_hashes = set()
    for name in CONFIGS:
        for year in YEARS:
            for seed in SEEDS:
                directory = OUT / 'runs' / f'{name}_y{year}_s{seed}'
                record = json.loads((directory / 'result.json').read_text())
                assert record['status'] == 'completed'
                assert record['max_steps'] == STEPS and record['reload_exact_match']
                assert record['gradient_train_end'] < record['validation_start']
                assert record['best_step'] > 0
                trajectories = json.loads((directory / 'trajectories.json').read_text())
                best = min(trajectories['validation'], key=lambda x: x['native_valid_loss'])
                assert best['step'] == record['best_step']
                source_hashes.add(record['source_sha256'])
                for policy in ['last', 'best']:
                    c = record[policy]['checks']
                    assert c['rows'] == 6660 and c['finite'] and c['quantiles_monotone']
                    assert c['metric_matches_library_after_pooling']
                    checks.append(dict(run=record['run_id'], policy=policy, **c))
    assert len(source_hashes) == 1, 'Training used more than one script version'
    assert sha256(ROOT / 'pmm_robustness.py') in source_hashes
    # Same seed-1, final-state experiment as the existing October run.
    old_map = {'GMM': 'GMM_fixed_formal', 'PMM_fixed': 'PMM_B_extension',
               'PMM_weighted': 'PMM_C_extension'}
    reproduction = []
    for name, old_name in old_map.items():
        new = pd.read_parquet(OUT / 'runs' / f'{name}_y2015_s1' / 'last_validation_predictions.parquet')
        old = pd.read_parquet(ROOT / 'experiment_results' / f'{old_name}_validation_predictions.parquet')
        old = old.rename(columns=lambda c: c.replace('.0', ''))
        pair = new.merge(old, on=['unique_id', 'ds'], validate='one_to_one', suffixes=('_new', '_old'))
        assert len(pair) == 6660
        cols = ['NHITS', 'NHITS-median'] + QCOLS
        error = max(float(abs(pair[c + '_new'] - pair[c + '_old']).max()) for c in cols)
        reproduction.append(dict(model=name, max_abs_prediction_difference=error,
                                 equal_within_1e_minus_6=error < 1e-6))
    test_checks = []
    for path in sorted((OUT / 'test').glob('*_checks.json')):
        check = json.loads(path.read_text())
        assert check['rows'] == 6660 and check['finite'] and check['quantiles_monotone']
        assert check['metric_matches_library_after_pooling']
        test_checks.append(dict(file=path.name, **check))
    assert len(test_checks) == 6
    result = dict(preserved_files=preserved, completed_runs=18,
                  same_source_sha256=next(iter(source_hashes)),
                  all_reload_exact=True, predictions_checks=checks,
                  test_predictions_checks=test_checks,
                  october_reproduction=reproduction)
    write_json(OUT / 'verification.json', result)
    return result


def plots(validation, selection, test):
    colors = {'GMM': '#3478b8', 'PMM_fixed': '#d47743', 'PMM_weighted': '#318464'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, year in zip(axes, YEARS):
        for i, name in enumerate(CONFIGS):
            for policy, offset, marker in [('last', -0.13, 'o'), ('best', 0.13, 's')]:
                part = validation.query('name == @name and year == @year and policy == @policy')
                ax.scatter(i + offset + np.linspace(-.04, .04, len(part)), part.pooled_scrps,
                           color=colors[name], marker=marker, s=44, alpha=.8)
                ax.plot([i + offset - .09, i + offset + .09], [part.pooled_scrps.mean()] * 2,
                        color='black', lw=2)
        ax.set_xticks(range(3), ['GMM', 'PMM fixed', 'PMM learned'])
        ax.set_title(f'{year} validation | 3 training seeds')
        ax.set_ylabel('Pooled sCRPS (lower is better)')
        ax.grid(axis='y', alpha=.2)
    axes[0].scatter([], [], color='gray', marker='o', label='Final step')
    axes[0].scatter([], [], color='gray', marker='s', label='Best native validation')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'validation_comparison.png', dpi=180)
    plt.close(fig)

    y, s, tags, _ = load_local_data()
    truth = y[(y.unique_id == 'TotalAll') & (y.ds.dt.year == 2016)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, chosen in zip(axes, [selection['GMM_baseline'], selection['selected_PMM']]):
        name, policy = chosen['name'], chosen['policy']
        pred = pd.read_parquet(OUT / 'test' / f'{name}_{policy}_s1_predictions.parquet')
        part = pred[pred.unique_id == 'TotalAll']
        ax.fill_between(part.ds, part['NHITS-lo-90'], part['NHITS-hi-90'],
                        alpha=.18, color=colors[name], label='90% interval')
        ax.plot(part.ds, part.NHITS, color=colors[name], label='Forecast mean', lw=2)
        ax.plot(truth.ds, truth.y, 'o-', color='#222222', label='Actual', ms=3)
        ax.set_title(f'{name} | {policy} | fixed training seed 1')
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel('Country total (original data units)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('2016 previously inspected test period — exploratory comparison', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'country_test_intervals.png', dpi=180)
    plt.close(fig)


def build_report():
    validation = pd.read_csv(OUT / 'validation_runs.csv')
    summary = pd.read_csv(OUT / 'validation_summary.csv')
    test = pd.read_csv(OUT / 'test_runs.csv')
    selection = json.loads((OUT / 'selection.json').read_text())
    assert len(validation) == 36 and len(test) == 6 and selection['test_evaluated']
    checks = verification()
    audit = json.loads((OUT / 'data_audit.json').read_text())
    paired = validation.pivot(index=['year', 'seed', 'policy'], columns='name', values='pooled_scrps').reset_index()
    paired['weighted_relative_improvement'] = 1 - paired.PMM_weighted / paired.PMM_fixed
    paired['weighted_better_than_GMM'] = paired.PMM_weighted < paired.GMM
    paired.to_csv(OUT / 'paired_comparison.csv', index=False)
    delta = validation.pivot(index=['name', 'year', 'seed'], columns='policy', values='pooled_scrps').reset_index()
    delta['best_minus_last'] = delta['best'] - delta['last']
    delta.to_csv(OUT / 'checkpoint_comparison.csv', index=False)
    checkpoint_summary = delta.groupby('name').agg(
        mean_best_minus_last=('best_minus_last', 'mean'),
        best_improves_runs=('best_minus_last', lambda x: int((x < 0).sum())),
        total_runs=('best_minus_last', 'count')).reset_index()
    by_year = validation.groupby(['year', 'name', 'policy']).pooled_scrps.agg(['mean', 'std']).reset_index()
    by_year.to_csv(OUT / 'validation_by_year.csv', index=False)
    test_summary = test.groupby(['name', 'policy']).agg(
        runs=('pooled_scrps', 'count'), mean_scrps=('pooled_scrps', 'mean'),
        std_scrps=('pooled_scrps', 'std'), min_scrps=('pooled_scrps', 'min'),
        max_scrps=('pooled_scrps', 'max'), mean_mae=('mae', 'mean'),
        mean_coverage90=('coverage90', 'mean')).reset_index()
    test_summary.to_csv(OUT / 'test_summary.csv', index=False)
    level_rows = []
    for r in validation.itertuples():
        m = pd.read_csv(OUT / 'runs' / f'{r.name}_y{r.year}_s{r.seed}' / f'{r.policy}_validation_metrics.csv')
        m['name'], m['year'], m['seed'], m['policy'] = r.name, r.year, r.seed, r.policy
        level_rows.append(m)
    levels = pd.concat(level_rows, ignore_index=True)
    levels.to_csv(OUT / 'validation_by_level.csv', index=False)
    bias_table = levels[(levels.level.isin(['Country', TAG_NAMES[-1]])) & (levels.policy == 'last')]
    bias_table = bias_table.groupby(['name', 'level'])[['bias', 'coverage90', 'width90']].mean().reset_index()
    metric_examples = validation[(validation.year == 2015) & (validation.seed == 1) &
                                 (validation.policy == 'last')][
        ['name', 'pooled_scrps', 'native_series_scaled_crps_mean']]
    y, s, _, _ = load_local_data()
    zero_rows = []
    series_score_rows = []
    for year in YEARS:
        truth = y[y.ds.dt.year == year]
        totals = truth.groupby('unique_id', observed=True).y.sum()
        zero_ids = totals[totals == 0].index
        for name in CONFIGS:
            for seed in SEEDS:
                for policy in ['last', 'best']:
                    pred = pd.read_parquet(OUT / 'runs' / f'{name}_y{year}_s{seed}' /
                                           f'{policy}_validation_predictions.parquet')
                    joined = truth[['unique_id', 'ds', 'y']].merge(
                        pred, on=['unique_id', 'ds'], validate='one_to_one')
                    series_scores = scaled_crps(joined, models={'NHITS': QCOLS}, quantiles=EVAL_Q)
                    positive = series_scores[~series_scores.unique_id.isin(zero_ids)]
                    zero_part = joined[joined.unique_id.isin(zero_ids)]
                    series_score_rows.append(dict(
                        name=name, year=year, seed=seed, policy=policy,
                        positive_total_series=len(positive), zero_total_series=len(zero_ids),
                        mean_scrps_positive_total_series=float(positive.NHITS.mean()),
                        median_scrps_positive_total_series=float(positive.NHITS.median()),
                        mae_zero_total_series=float((zero_part.NHITS - zero_part.y).abs().mean())))
                    for uid, part in pred[pred.unique_id.isin(zero_ids)].groupby('unique_id', observed=True):
                        zero_rows.append(dict(name=name, year=year, seed=seed, policy=policy,
                                              unique_id=uid, actual_total=0.,
                                              predicted_mean=float(part.NHITS.mean()),
                                              predicted_max=float(part.NHITS.max()),
                                              mean_upper90=float(part['NHITS-hi-90'].mean())))
    pd.DataFrame(zero_rows).to_csv(OUT / 'zero_series_diagnostics.csv', index=False)
    series_diagnostics = pd.DataFrame(series_score_rows)
    series_diagnostics.to_csv(OUT / 'per_series_score_diagnostics.csv', index=False)
    positive_examples = series_diagnostics[(series_diagnostics.year == 2015) &
        (series_diagnostics.seed == 1) & (series_diagnostics.policy == 'last')][
        ['name', 'mean_scrps_positive_total_series', 'median_scrps_positive_total_series',
         'mae_zero_total_series']]
    pmm = selection['selected_PMM']
    pmm_cfg = CONFIGS[pmm['name']]
    fixed_last = summary.query("name == 'PMM_fixed' and policy == 'last'").iloc[0]
    weighted_last = summary.query("name == 'PMM_weighted' and policy == 'last'").iloc[0]
    reduction = 1 - weighted_last.mean_scrps / fixed_last.mean_scrps
    win_count = int((paired.query("policy == 'last'").PMM_weighted < paired.query("policy == 'last'").PMM_fixed).sum())
    plots(validation, selection, test)
    report = f'''# PMM 追加檢查結果（2026-11-08）

已完成 **18 次正式訓練**：3 個設定 × 3 個訓練種子 × 2 個驗證年度。每次保存最後權重與內部驗證最佳權重，共 36 組驗證預測；選擇完成後，再計算 6 組 2016 預測。這些是本輪實際重訓結果，不是把原有分數重新排版。

依事先設定的平均驗證 pooled sCRPS，選出 **{pmm['name']} / {pmm['policy']}**，平均驗證分數 **{pmm['mean_validation_scrps']:.6f}**。用最後權重作直接對照，PMM 加入可學習混合權重後，六組平均驗證誤差降低 **{reduction:.1%}**，在 **{win_count}/6** 個年度／訓練種子組合較低。

完整設定見 [追加檢查計畫](PMM_追加檢查計畫.md)，實際執行程式為 [pmm_robustness.py](pmm_robustness.py)，較精簡的使用入口為 [PMM_改善版.ipynb](PMM_改善版.ipynb)。原始兩個 notebook 與 10 月 29 日的比較／選擇檔案雜湊保持不變。

## 最值得改善的地方

### 1. 階層順序必須正確

本機確認原始序列順序與 S 不一致。程式改用 S 的原始索引作為 ordered categorical 類別，並檢查訓練後的 `nf.uids`；S 最後 304 列維持單位矩陣。平均預測必須符合 `all = S @ bottom`。

不能把 S 單獨按字母排序，也不能把各序列的分位數直接加總。HINT 先對抽樣結果作 BottomUp，再重新計算分位數。

### 2. PMM 的混合權重比只降低學習率更值得檢查

固定權重與可學習權重 PMM 均使用 learning_rate=0.0003，其他訓練設定相同。`weighted=True` 讓每個預測位置的各 Poisson 成分機率可由網路學習；固定權重時每個成分占 10%。它同時增加了網路的權重輸出，因此這是完整配置的對照，不能宣稱把所有差異都獨立歸因於一個機制。

下表是兩個年度、三個訓練種子的平均與標準差；**數字越低越好**。標準差混合了年度差異與訓練波動，不是統計信賴區間。

{md_table(summary[['name', 'policy', 'runs', 'mean_scrps', 'std_scrps', 'min_scrps', 'max_scrps']])}

![驗證比較](pmm_checks_20261108/validation_comparison.png)

各驗證年度分開比較：

{md_table(by_year)}

### 3. 早停不等於自動還原最佳模型

原程式沒有還原最佳 checkpoint；本輪在訓練 callback 保存內部 `ptl/val_loss` 最低的權重，再與最後權重比較。排除訓練前 sanity check 的第 0 步。下表差值為 best 減 last，負值代表還原內部最佳權重後，外部 pooled sCRPS 改善。

`early_stop_patience_steps=10` 在每 50 步檢查一次的設定下，代表 10 次驗證未改善，通常約為最佳步驟之後 500 個訓練步驟；不是只等待 10 個梯度更新。每次實際步數與最佳步數保存在 `validation_runs.csv`。

{md_table(checkpoint_summary)}

內部早停看的是階層整合前的指標，外部評估是 BottomUp 後的 pooled 指標，因此內部最佳 checkpoint 不保證最後表現最佳。本輪依完整驗證結果選擇使用 `{pmm['policy']}`，沒有直接假定 best 必勝。

### 4. 檢查誤差在哪個階層累積

以下是最後權重在六組驗證的平均診斷。bias 為預測減真值，正值表示高估；coverage90 為真值落在 90% 預測區間內的比例。

{md_table(bias_table, digits=4)}

這份階層有 304 條底層序列。BottomUp 下，全國的平均偏差等於底層平均偏差的 304 倍；因此每條底層看似不大的系統性高估，會累積成全國層級的大幅高估。

Country 每個年度只有 12 個月，單一年度的覆蓋率會以 1/12 為單位跳動，應搭配多年度與區間寬度解讀。

## 2016 結果：設定選定後的探索性檢查

2016 在 10 月的實驗中已經看過，**不是新的獨立盲測**。本輪選擇只依 2014、2015 驗證平均，不挑測試分數最好的種子，也沒有依本輪測試分數更換設定。各模型使用在 2015 fold 訓練的權重，以截至 2015-12 的歷史預測 2016；沒有把 2016 目標用於訓練。

{md_table(test_summary[['name', 'policy', 'runs', 'mean_scrps', 'std_scrps', 'min_scrps', 'max_scrps', 'mean_mae']])}

{md_table(test[['name', 'policy', 'seed', 'pooled_scrps', 'mae', 'coverage90']])}

圖示固定使用事先指定的訓練種子 1，不挑最好看的預測。

![全國測試預測](pmm_checks_20261108/country_test_intervals.png)

## 指標與資料適用性

- 主指標為 `2 * sum(pinball loss) / (98 * sum(abs(y)))`，按同一階層所有序列與月份共同加總；使用 0.01–0.49、0.51–0.99 共 98 個分位數，與上一輪外部定義一致。
- 每份驗證與測試預測都以 utilsforecast 合併 ID 後的計算結果交叉核對主指標。另保留 `native_series_scaled_crps_mean`，它先算逐序列比例再平均，兩者權重意義不同，不能混稱同一個數字。
- 2014 有 6 條全零驗證序列，2015 有 1 條。逐序列指標的 epsilon 分母會讓某些非零預測產生極大分數；這不是預測值本身一定同樣巨大。
- 資料的 {audit['fractional_fraction']:.2%} 是非整數。PMM 程式允許非負小數，但 Poisson 本來是離散計數模型；能算 loss 不代表生成假設吻合。沒有擅自四捨五入或改變資料單位。
- Poisson 的單一成分平均數與變異數相同；混合後可以過度離散，不能用單一成分的限制概括整個 PMM。
- 季節性 naive 的指標另存 `seasonal_naive_2014.csv`、`seasonal_naive_2015.csv`。它的點預測有實際比較意義，但零寬區間只是退化分布，不是有校準能力的機率基準。

同一份 2015、seed=1、最後權重的預測，兩種彙總方式會得到以下不同數字。這不是重新調模型或挑分數；兩個指標都保留。全零序列的實際預測另存 `zero_series_diagnostics.csv`。

{md_table(metric_examples)}

另將分母有定義的序列與全零序列分開診斷：前者列出逐序列 sCRPS 的平均／中位數，後者列原資料單位的 MAE。這個表沒有拿來重新選模型，也沒有把全零序列從主要 pooled 指標刪掉。

{md_table(positive_examples)}

## 可靠性與限制

- 18 次訓練均完整執行到 2000 步或共同早停條件；無時間中斷的訓練混入比較。
- 所有訓練使用同一份程式版本；每次都確認最佳權重重載可完全重現預測。
- 所有預測通過列數、ID／月份對齊、finite、分位數單調性、階層平均加總、評分公式等檢查，詳見 `pmm_checks_20261108/verification.json`。
- 三個訓練種子為 1、7、21；預測抽樣種子固定為 1。這次確實重新訓練，不是只改抽樣種子。
- 本輪 PMM 都使用 0.0003，沒有重新搜尋學習率、輸入長度或成分數，因此不能把選定設定稱為全域最佳參數。
- 兩個年度共享訓練歷史；三個種子、單一資料集仍不足以證明模型在所有資料上較好。沒有把相關的六組結果當作六個獨立資料集做顯著性宣稱。
- 本輪最佳 checkpoint 的判定仍採套件內部 sCRPS。若未來要改為直接監控整合後的 pooled 指標，需要另外設計實驗，不能把指標尺度或計算方式悄悄改掉。

本輪對原有 seed=1、2015 最後權重預測的重現差異：

{md_table(pd.DataFrame(checks['october_reproduction']))}

## 使用選定的 PMM

`PMM_改善版.ipynb` 會讀取本輪 selection，顯示設定並載入事先固定種子 1 的權重。預設不重新跑 18 次訓練；可直接重現選定模型的 2016 預測。

選定 PMM 的核心設定：`learning_rate={pmm_cfg['lr']}`、`weighted={pmm_cfg['weighted']}`、`input_size=24`、`n_components=10`、`scaler_type='robust'`、權重策略 `{pmm['policy']}`。完整資料排序、權重重載與日期產生邏輯均在實際執行模組中。

如果下一階段還要調 `input_size` 或成分數，應另設驗證搜尋並保留後續未看過的時間區間；本輪的 2016 分數不適合反覆用來挑參數。
'''
    (ROOT / 'PMM_追加檢查結果.md').write_text(report)
    make_notebook(selection)
    print('REPORT_WRITTEN', ROOT / 'PMM_追加檢查結果.md', flush=True)


def make_notebook(selection):
    def markdown(source):
        return dict(cell_type='markdown', metadata={}, source=source.splitlines(keepends=True))
    def code(source):
        return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                    source=source.splitlines(keepends=True))
    chosen = selection['selected_PMM']
    cells = [
        markdown(f'''# PMM 改善版：檢查結果與可重現預測

本輪完成 18 次訓練，以 2014、2015 驗證結果選出 **{chosen['name']} / {chosen['policy']}**。
請從 `pmm` 資料夾開啟，選擇專案 `.venv` Python 核心。

本 notebook 預設讀取已完成結果並重載模型，**不重新訓練**。原始 `hierarchical_forecasting.ipynb` 與 `ＰＰＰ.ipynb` 保留。
2016 曾在前輪實驗中看過，因此這裡是重現／探索性比較，不能視為新的盲測。
完整細節見 [追加檢查結果](PMM_追加檢查結果.md)。
'''),
        code('''# 先設定專案內的執行緒、暫存與快取環境，再載入其他套件。
from pmm_robustness import (
    ROOT, OUT, CONFIGS, make_base, load_local_data, forecast,
)
import json
import pandas as pd
import torch
from IPython.display import display, Image
selection = json.loads((OUT / 'selection.json').read_text())
chosen = selection['selected_PMM']
print('驗證集選出的 PMM：', chosen)
print('模型設定：', CONFIGS[chosen['name']])
'''),
        markdown('''## 六組驗證結果

每個設定均使用 3 個真正重新訓練的種子與 2 個驗證年度。`mean_scrps` 越低越好。
`last` 是最後一步；`best` 是內部驗證指標最佳一步。內部指標與最終階層整合後的指標不同，所以 `best` 不保證更好。
'''),
        code('''display(pd.read_csv(OUT / 'validation_summary.csv'))
display(Image(filename=str(OUT / 'validation_comparison.png')))
'''),
        markdown('''## 載入選定模型並重現預測

使用事先固定的訓練種子 1，以 2015 年底以前的歷史預測 2016；不挑測試分數最好的種子。
資料會先對齊 S 的順序；預測日期由歷史最後月份推算，不從待預測的真值取得。
'''),
        code('''training_seed = 1
Y, S, tags, audit = load_local_data()
history = Y[Y['ds'].dt.year < 2016].copy()
model_dir = OUT / 'runs' / f"{chosen['name']}_y2015_s{training_seed}"
base = make_base(CONFIGS[chosen['name']], training_seed)
base.trainer_kwargs['callbacks'] = []
base.load_state_dict(torch.load(
    model_dir / f"{chosen['policy']}_weights.pt", weights_only=True
))
prediction = forecast(base, history, S, sampling_seed=1)
display(prediction[['unique_id', 'ds', 'NHITS', 'NHITS-lo-90', 'NHITS-hi-90']].head(12))
'''),
        markdown('''## 已記錄的 2016 結果

這是設定選定後才計算的本輪結果，但測試年度已在前輪看過。請同時看三個訓練種子的平均與範圍。
'''),
        code('''display(pd.read_csv(OUT / 'test_summary.csv'))
display(Image(filename=str(OUT / 'country_test_intervals.png')))
'''),
        markdown('''## 下一次修改時保留的檢查

1. `unique_id` 順序必須與 S 一致；S 的底層區塊維持單位矩陣。
2. 確認載入的是選定策略的權重，不能把 early stopping 當成自動還原最佳權重。
3. 保留時間順序切分；只依驗證集選設定。
4. 同時檢查 pooled sCRPS、點預測誤差、各階層偏差與區間覆蓋率。
5. 多個訓練種子要重新訓練；只改預測抽樣種子不能代替。

完整重跑入口見 `PMM_追加檢查計畫.md`。程式會略過已完成的訓練；不同計畫應另設結果目錄。
'''),
    ]
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='PMM experiment (.venv Python 3.10)', language='python', name='pmm-experiment'),
        language_info=dict(name='python', version='3.10')), nbformat=4, nbformat_minor=5)
    for i, cell in enumerate(cells):
        cell['id'] = f'pmm-check-{i:02d}'
    (ROOT / 'PMM_改善版.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    build_report()
