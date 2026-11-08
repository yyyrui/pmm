以下針對原本 `hierarchical_forecasting.ipynb` 複製後的教學程式，依序替換六個位置。使用專案 `.venv` 核心，略過原本的 `pip install` 格，先重新啟動核心，再由上往下執行。

目前 `ＰＰＰ.ipynb` 下半部的教學片段是 Markdown 參考文字；若在該處貼上以下程式，請改成 **Code cell**。上方整套實驗程式不需要和這六段同時執行。

**1. 匯入套件：把 GMM 改成 PMM，補上 callback 與 torch。放在第一個 Code cell。**

```python
import os
from pathlib import Path

# 在 pmm 資料夾執行；暫存、快取都留在專案內。
project = Path.cwd()
assert (project / 'data/hierarchical/TourismLarge/data.csv').exists()
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
for key, folder in {
    'TMPDIR': '.pmm_manual_runtime/tmp',
    'MPLCONFIGDIR': '.pmm_manual_runtime/matplotlib',
    'XDG_CACHE_HOME': '.pmm_manual_runtime/cache',
    'TORCH_HOME': '.pmm_manual_runtime/torch',
}.items():
    path = project / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import logging
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback
from datasetsforecast.hierarchical import HierarchicalData
from hierarchicalforecast.utils import aggregate, HierarchicalPlot
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, HINT
from neuralforecast.losses.pytorch import PMM, sCRPS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import augment_calendar_df
from utilsforecast.plotting import plot_series

torch.set_num_threads(1)
```

**2. 資料讀取與切分：把 `# Load hierarchical dataset` 和 `# Train test splits` 兩段一起替換。**

排序必須在切出 `Y_train_df`、`Y_test_df` 之前完成。

```python
Y_df, S_df, tags = HierarchicalData.load('./data', 'TourismLarge')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
Y_df, _ = augment_calendar_df(df=Y_df, freq='MS')

if 'unique_id' not in S_df.columns:
    S_df = S_df.reset_index(names='unique_id')

hierarchy_ids = S_df['unique_id'].tolist()
assert set(Y_df['unique_id']) == set(hierarchy_ids)
Y_df['unique_id'] = Y_df['unique_id'].astype(
    pd.CategoricalDtype(categories=hierarchy_ids, ordered=True)
)
Y_df = Y_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

S = S_df.drop(columns='unique_id').to_numpy()
assert list(TimeSeriesDataset.from_df(Y_df)[1]) == hierarchy_ids
assert list(S_df.drop(columns='unique_id').columns) == hierarchy_ids[-S.shape[1]:]
assert np.allclose(S[-S.shape[1]:], np.eye(S.shape[1]))
assert np.isfinite(Y_df['y']).all() and (Y_df['y'] >= 0).all()

horizon = 12
Y_test_df = Y_df.groupby('unique_id', observed=True).tail(horizon).copy()
Y_train_df = Y_df.drop(Y_test_df.index).copy()
```

刪除原本在 `nf.fit()` 前的 `Y_df['y'] = Y_df['y'] * (Y_df['y'] > 0)`。本資料沒有負值；若未來出現負值，上面的檢查會先攔下，應查明資料原因，不要直接四捨五入或改成零。

**3. 在建立 NHITS 之前，新增保存最佳權重的 Code cell。**

`early_stop_patience_steps` 只負責停止訓練，還需要明確保存和還原權重。這裡把最佳權重保存在記憶體中。

```python
class KeepBestWeights(Callback):
    def __init__(self):
        self.best_loss = float('inf')
        self.best_step = None
        self.best_state = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        value = float(trainer.callback_metrics['ptl/val_loss'])
        if not np.isfinite(value):
            raise RuntimeError('驗證 loss 不是有限數值')
        if value < self.best_loss:
            self.best_loss = value
            self.best_step = int(trainer.global_step)
            self.best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in pl_module.state_dict().items()
            }
```

**4. 把原本 `nhits = NHITS(...)`、`model = HINT(...)` 那一整格替換成以下內容。**

```python
quantiles = np.arange(1, 100) / 100

nhits = NHITS(
    h=horizon,
    input_size=24,
    loss=PMM(
        n_components=10,
        quantiles=quantiles,
        weighted=True,        # 新增：讓混合成分的機率可以學習
        num_samples=1000,
    ),
    hist_exog_list=['month'],
    max_steps=2000,
    early_stop_patience_steps=10,
    val_check_steps=50,
    scaler_type='robust',
    learning_rate=0.0003,     # 保留你修改後的學習率
    valid_loss=sCRPS(quantiles=quantiles),
    random_seed=1,
    callbacks=[KeepBestWeights()],
    accelerator='cpu',
    devices=1,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)

model = HINT(
    h=horizon,
    S=S,
    model=nhits,
    reconciliation='BottomUp',
    alias='NHITS',            # 保持後面畫圖、評分使用的欄位名稱
)
```

`input_size=24`、`n_components=10`、`max_steps=2000`、`robust`、`BottomUp` 都保留。這次有證據支持的核心模型修改是 `weighted=True`，沒有重新搜尋其他學習率或輸入長度。

**5. 把原本包含 `nf.fit()`、`nf.predict()` 的那一格整段替換。**

```python
nf = NeuralForecast(models=[model], freq='MS')
nf.fit(df=Y_train_df, val_size=12)
assert list(nf.uids) == hierarchy_ids

# NeuralForecast 會複製模型，所以要從「訓練後的模型」取得 callback。
trained = nf.models[0].model
best = next(
    cb for cb in trained.trainer_kwargs['callbacks']
    if isinstance(cb, KeepBestWeights)
)
assert best.best_state is not None
trained.load_state_dict(best.best_state)
trained.trainer_kwargs['callbacks'] = []
print('已還原最佳權重，步數：', best.best_step)

np.random.seed(1)
Y_hat_df = nf.predict(random_seed=1)
Y_hat_df = Y_hat_df.rename(columns=lambda c: c.replace('.0', ''))
```

`Y_train_df` 最後 12 個月由 `val_size=12` 留作驗證；`Y_test_df` 不傳入 `fit()`。原本的 `plot_series(...)` 可以沿用。

**6. 最後評分：把 `df_metrics = ...` 到 `metrics` 的整格替換。**

這項修改是修正比較方式，不會讓模型本身變準。全零序列會讓逐序列 scaled CRPS 的分母接近零，因此另列 pooled sCRPS；不能把它和舊版的逐序列平均當成同一個指標。

```python
df_metrics = Y_test_df[['unique_id', 'ds', 'y']].merge(
    Y_hat_df, on=['unique_id', 'ds'], validate='one_to_one'
)
models_to_score = ['NHITS']

# 若已執行原教學的 AutoARIMA / reconciliation，取消下列兩行的註解：
# df_metrics = df_metrics.merge(Y_rec_df, on=['unique_id', 'ds'], validate='one_to_one')
# models_to_score.append('AutoARIMA')

assert len(df_metrics) == len(Y_test_df)

levels = np.arange(2, 100, 2)
eval_q = np.r_[np.arange(1, 50), np.arange(51, 100)] / 100
rows = []
for level_name, ids in {**tags, 'Overall': hierarchy_ids}.items():
    part = df_metrics[df_metrics['unique_id'].isin(ids)]
    denominator = part['y'].abs().sum()
    row = {'level': level_name, 'metric': 'pooled_scaled_crps'}
    for name in models_to_score:
        columns = (
            [f'{name}-lo-{lv}' for lv in levels[::-1]]
            + [f'{name}-hi-{lv}' for lv in levels]
        )
        error = part['y'].to_numpy()[:, None] - part[columns].to_numpy()
        pinball = np.maximum(eval_q * error, (eval_q - 1) * error)
        row[name] = (
            2 * pinball.sum() / (len(eval_q) * denominator)
            if denominator > 0 else np.nan
        )
    rows.append(row)

metrics = pd.DataFrame(rows)
metrics
```

最主要的三個實作修改是：**對齊 S 排序、PMM 加 `weighted=True`、訓練後還原最佳權重**。學習率維持 `0.0003`。評分的變更要另外標示為 pooled sCRPS。
