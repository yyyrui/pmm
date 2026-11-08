"""Build a Chinese report/notebook outputs from recorded executions, not invented scores."""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import nbformat

root=Path('.')
out=root/'experiment_results'
def read(name):
    p=out/name
    return json.loads(p.read_text()) if p.exists() else {}
def fmt(x):
    return '—' if x is None else f'{x:.6f}'
session=read('session.json')
audit=read('data_audit.json')
original=read('original_gmm_attempt.json')
selection=read('selection.json')
isolation=read('ordering_isolation.json')
records=[]
for phase in ['short','formal','extension']:
    for name in ['GMM_fixed','PMM_A','PMM_B','PMM_C']:
        rec=read(f'{name}_{phase}.json')
        if rec: records.append(rec)
rows=[]
for r in records:
    rows.append({'run':r['name'],'phase':r['phase'],'status':r.get('status','in_progress'),
        'actual_steps':r.get('actual_steps',r.get('observed_progress_steps')),
        'validation_pooled_scrps':r.get('validation_scrps'),'test_pooled_scrps':r.get('test_scrps'),
        'learning_rate':r['lr'],'weighted':r.get('weighted',False),'timed_out':r.get('timed_out',False)})
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
table='| 版本 | 階段 | 實際步數 | 狀態 | 驗證 pooled sCRPS | 測試 pooled sCRPS |\n|---|---|---:|---|---:|---:|\n'
for r in rows:
    table+=f"| {r['run']} | {r['phase']} | {r['actual_steps'] or '—'} | {r['status']} | {fmt(r['validation_pooled_scrps'])} | {fmt(r['test_pooled_scrps'])} |\n"
now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')
completed={}
for rec in records:
    if rec['phase']!='short' and rec.get('status')=='completed':
        completed[rec['name']]=rec
pd.DataFrame([{k:r.get(k) for k in ['name','phase','lr','weighted','actual_steps','validation_scrps','test_scrps']}
              for r in completed.values()]).to_csv(out/'comparison_completed.csv',index=False)
calibration_table='| 版本 | 全國平均偏差 | 全國 90% 覆蓋率 | 底層平均偏差 | 底層 90% 覆蓋率 |\n|---|---:|---:|---:|---:|\n'
for name,rec in completed.items():
    dpath=out/f"{name}_{rec['phase']}_validation_diagnostics.csv"
    if dpath.exists():
        diag=pd.read_csv(dpath).set_index('level')
        country=diag.loc['Country'];bottom=diag.loc['Country/State/Zone/Region/Purpose']
        calibration_table+=f"| {name} | {country['bias']:.2f} | {country['coverage90']:.1%} | {bottom['bias']:.2f} | {bottom['coverage90']:.1%} |\n"
selected_rec=completed.get(selection.get('selected'),{})
completed_pmm_names=[name for name in ['PMM_A','PMM_B','PMM_C'] if name in completed]
sampling=read('sampling_sensitivity_summary.json')
sampling_text='預測抽樣敏感度尚未完成。'
if sampling:
    sampling_text='固定同一組訓練權重，預先指定預測抽樣 seed=1,2,3,4,5；不重訓、不挑最好 seed，也不改主表結果。'
    for run,stat in sampling['summary'].items():
        sampling_text+=f" `{run}` 的測試 pooled sCRPS 平均 {stat['mean']:.6f}、標準差 {stat['std']:.6f}、範圍 {stat['min']:.6f}–{stat['max']:.6f}。"
    sampling_text+=f" PMM 分數較低的抽樣 seed 有 {sampling['PMM_lower_score_inference_seed_count']}/5 個。這只量測預測抽樣波動，不是不同訓練種子的穩定性，也不是統計顯著性檢定。"
weight_findings='可學習權重設定尚未完成。'
if 'PMM_C' in completed and 'PMM_B' in completed:
    b,c=completed['PMM_B'],completed['PMM_C']
    improvement=(1-c['validation_scrps']/b['validation_scrps'])*100
    weight_findings=(f"在相同 learning_rate=0.0003、相同 split、seed 與早停規則下，B（固定權重）驗證分數 {b['validation_scrps']:.6f}，C（可學習權重）{c['validation_scrps']:.6f}，降低約 {improvement:.1f}%。"
      "B/C 只改 weighted 旗標；該旗標也會增加權重輸出頭，並非相同網路輸出維度。這是本次最直接的改善證據，支持固定等權重的限制是重要因素，但單一訓練種子不能證明唯一原因或跨資料集的一般結論。")
extension_note=('使用者於第一輪結果完成後追加 30 分鐘，延長截止至 14:03:57。第二、第三組仍是第一輪已記錄的設定，沒有新增第四組。第一輪測試結果已公開，因此延伸測試屬於同一測試期的探索性比較，**不是新的獨立確認集**。設定選擇仍只使用 2015 驗證集。第一輪 notebook、報告與選擇紀錄保留在 `experiment_results/round1/`。'
                if session.get('extension_minutes') else '本輪沒有延伸實驗。')
report=f'''# GMM → PMM 實驗紀錄

產生時間：{now}。本次總時限：2026-10-29 12:33:57 至 {session['deadline']}（台北時間，包含安裝、環境排錯與中斷等待）。

{extension_note}

目前驗證集選定 `{selection.get('selected','尚未完成')}`。它的驗證 pooled sCRPS = {fmt(selected_rec.get('validation_scrps'))}，測試 pooled sCRPS = {fmt(selected_rec.get('test_scrps'))}；修正排序的 GMM 分別為 {fmt(completed.get('GMM_fixed',{}).get('validation_scrps'))}、{fmt(completed.get('GMM_fixed',{}).get('test_scrps'))}。數值越低越好。各設定的完整數字與限制如下。

原檔 `hierarchical_forecasting.ipynb` 未修改；修改版為 `ＰＰＰ.ipynb`（全形英文字母）。原檔 SHA256：`{session['original_sha256']}`。

## 用途、環境與資料

原程式用 NHITS 產生機率預測，再由 HINT BottomUp 加總成一致的階層預測。GMM 是 Gaussian mixture 損失；PMM 是 Poisson mixture 損失，不是另一個獨立預測網路。兩者 API 可接入相同 NHITS/HINT。

使用專案 `.venv` 的 Python 3.10、CPU 單執行緒；完整版本見 `experiment_results/requirements-lock.txt`。NeuralForecast 3.3.0、PyTorch 2.14.1、HierarchicalForecast 1.5.3、UtilsForecast 0.2.17、DatasetsForecast 1.0.2。系統 Python 3.15 沒有套件，因此另建環境。原 notebook 未鎖版本，無法重建當年的精確環境。

TourismLarge：555 條月序列、228 個月、126,540 列，S 為 555 × 304。資料下載來源是 Nixtla 公開資料集；檔案雜湊見 `data_audit.json`。

| 資料用途 | 時段 | 每條序列月份 |
|---|---|---:|
| 梯度訓練 | 1998-01 至 2014-12 | 204 |
| 驗證、早停與設定選擇 | 2015-01 至 2015-12 | 12 |
| 最終測試 | 2016-01 至 2016-12 | 12 |

與原程式一致：`fit(pretest, val_size=12)`；預測測試期時可用已知的 2015 歷史，但不把 2016 目標傳入訓練或選參數。沒有把小數四捨五入、沒有改資料單位。未重跑教學的 AutoARIMA/MinTrace 支線，因本次目標是 GMM 與 PMM。

## 每次修改與實際結果

1. 複製原 notebook、清除複本中的歷史輸出，將 GMM 介面替換為 PMM；原檔保持原樣。
2. 原樣執行 GMM 的資料／建模／訓練／預測 cells 7、8、13–17。第一次在 OpenMP 執行緒初始化失敗（exit 139），第二次解除沙盒後仍 exit 139。改用單執行緒環境後成功；不曾更改原 notebook 的模型程式。
3. PMM 最小試跑完成 2 步訓練及 6,660 列預測。GMM、PMM A 的 200 步診斷也完成，PMM B 的短跑在觀察到 50 步後中斷，以確保先完成正式驗證選擇、最後才測試。
4. 發現序列順序不符：NeuralForecast 依 ID 排序，S 則保留階層順序。修正版使用與 S 一致的 ordered categorical ID，並以斷言檢查。不能單純字母排序 S，因 HINT BottomUp 假設 S 的最後 304 列為底層單位矩陣。
5. 發現逐序列 sCRPS 的零分母問題，保留套件原指標，另記錄 pooled sCRPS；在正式訓練之前決定以後者選設定，沒有根據測試結果調整。
6. 正式預算沿用原設定最多 2,000 步、每 50 步驗證、patience 10。短跑是獨立重訓，不能當成完整訓練。原套件預設的學習率衰減取決於 max_steps，因此短跑與完整訓練也不是單純的續訓關係。
7. 延長時間後，B 因前次沒有權重／optimizer checkpoint，從 seed=1 重新訓練；C 依原計畫執行。每 250 步增加本機恢復 checkpoint，不改優化與最後權重的計分規則。中斷的歷史紀錄保留，完成結果另標 `extension`。

原樣 GMM：狀態 `{original.get('status')}`，實際 {original.get('actual_steps')} 步，耗時 {original.get('elapsed_seconds',0):.1f} 秒。其驗證 pooled sCRPS = {fmt(read('original_gmm_validation_summary.json').get('validation_pooled_scrps'))}。另固定同一組原始權重、不重訓，只修正推論順序，驗證 pooled sCRPS = {fmt(isolation.get('validation_pooled_scrps'))}（seed=1）。

共同設定：h=12、input_size=24、10 components、robust scaler、month 歷史特徵、seed=1、1,000 samples、99 個訓練／驗證分位數、原預設 NHITS 架構與 batch/window 設定。保留與原程式相同的最後訓練權重策略；原程式未儲存／還原驗證最佳 checkpoint。

| PMM 設定 | 學習率 | mixture weights | 動機 |
|---|---:|---|---|
| A：直接替換 | 0.001 | 固定等權重 | 測試只換損失的效果 |
| B | 0.0003 | 固定等權重 | A 短跑偏高估，檢查較小更新是否穩定 |
| C | 0.0003 | 可學習權重 | 與 B 比較，檢查固定每成分 10% 機率是否過度受限 |

{table}

`short` 僅為試跑；`formal`、`extension` 都要求原 2,000 步預算或原早停規則。`completed` 且少於 2,000 步表示依共同規則早停，並非時間不足。`time_limited` 則不能當成已完成完整預算，也不參與完成組的設定選擇。除了第一輪已看過的 A，延伸階段只對驗證集選中的 PMM 計算測試分數，未選中的設定留白。

目前選定：`{selection.get('selected','尚未選定')}`；選擇依據：`{selection.get('selection_metric','尚未完成')}`。最終測試是否完成：`{selection.get('test_evaluated',False)}`。

已完成的 PMM 設定：`{completed_pmm_names}`；未完成／排除紀錄：`{selection.get('excluded',{})}`。表格中的 B 中斷列為第一輪歷史，延伸完成的 B 記在另一列。原樣 GMM 最後測試 pooled sCRPS：{fmt(read('original_gmm_test_summary.json').get('test_pooled_scrps'))}，其排序錯誤仍保留，僅作問題重現。

以下為各組**驗證集**的階層偏差／覆蓋率，正偏差代表高估。整體覆蓋率可能掩蓋全國層級問題，因此分開檢查：

{calibration_table}

![驗證分數與全國層級預測](experiment_results/validation_comparison.png)

{weight_findings}

{sampling_text}

![測試期全國層級區間](experiment_results/test_country_intervals.png)

## 指標與數值檢查

主要比較值採 `pooled_scaled_crps = 2 × Σ pinball(y, q) / (98 × Σ |y|)`，加總範圍為該階層所有序列與 12 個月，98 個 q 與原評估的 level=2,4,…,98 對應（不含中位數）。這是以總絕對目標作分母的尺度化 CRPS 近似；不同於目前套件先除每條序列各自分母再平均。原 notebook 的歷史 Overall=0.122793 只是舊輸出，不能拿來當本次基準，也不能假定套件版本改變前後指標實作完全相同。

已額外確認：原 notebook 儲存的八個階層分數平均為 0.12279275，四捨五入後正好等於原 Overall。本次非負資料每層的目標總量相同，因此 pooled Overall 也等於八層 pooled 分數的平均。這支持指標彙總方式的對照，但無法識別當年的套件版本。另以簡單解析例子驗證目前 NeuralForecast 內部 sCRPS：理論值 0.1，套件回傳約 9.9（99 個分位數）。此共同倍率不改同一分位數設定下的排序或零 min_delta 早停，不是 PMM 相對較差的解釋；本次外部指標沒有這個倍率。

`*_metrics.csv` 保留原套件 scaled_crps、MAE、RMSE；`*_pooled_metrics.csv` 存穩定的共同指標。`*_diagnostics.csv` 提供各層 90% 區間覆蓋率、寬度、偏差與負平均預測數；`*_predictions.parquet` 保留實際預測。比較前檢查列數、ID／日期、finite 值、分位數單調性與階層平均值一致性。`verification.json` 另以套件在合併 ID 後的 scaled_crps 交叉驗證手算 pooled 公式。

## 已確認的事實

- 實際資料沒有負值，但 {audit.get('fractional_count')} 筆（{audit.get('fractional_fraction',0):.2%}）不是整數。原程式在切分後才歸零負值，對已切出的訓練資料不起作用；此資料沒有負值，所以本次不是造成差異的原因。
- PMM 程式可以接受非負小數；0.25 的 likelihood 與梯度試算均為有限值。套件把 Poisson support 放寬成 nonnegative，並以 gamma 函數延伸計算。因此「能執行」不代表這是符合資料生成過程的離散機率模型。
- 原程式的 ID 順序與 S 不符；固定同一組已訓練權重，只改推論順序即可大幅改善驗證分數，確認排序／階層對應有實質影響。
- 驗證集有全零序列，套件逐序列 scaled_crps 的 epsilon 分母會使短跑分數達 10^13 等級。原指標仍完整保留；兩版都用同一個 pooled 定義作主要比較。
- 修正後的預測平均值通過階層加總檢查；沒有把分位數誤當成可直接相加的量。
- 官方 PMM 支援 n_components、quantiles、num_samples、weighted 等參數；本次沒有修改套件原始碼。[Nixtla PMM 文件](https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html#poisson-mixture-mesh-pmm)
- 實測 C 的可學習權重與 B 的固定權重有明顯效果差異，見上方控制條件與數值。A/B 的偏高估經 BottomUp 累加至全國，C 已大幅縮小此偏差。

## 可能原因與尚未確認的部分

- 旅遊資料多為實數估計值，Poisson 的離散計數假設可能不匹配；不能單憑小數比例證明它是誤差主因。
- 每個 Poisson 成分的平均數與變異數綁定；混合後仍可以過度離散，不能簡化成「PMM 的變異數必等於平均數」。相較 GMM 的個別可學習標準差，PMM 可能需要不同的成分位置／權重才能表達尾端、零值及季節變動。
- 訓練期逐序列樣本變異數／平均數中位數約 {audit.get('train_variance_over_mean_median',0):.2f}。這包含季節性與時間變化，**不是條件過度離散的證明**。
- PMM A 短跑有偏高估現象；rate 下界與混合權重診斷見 `*_rate_diagnostics.json`。softplus 下界可能影響梯度，但本次僅檢查預測參數，不能斷言發生梯度消失或成分崩潰。
- A 完整訓練的 rate 成分觸及下界比例約 0.88%，且沒有任何驗證預測點的全部成分同時在下界，故目前沒有支持「整個模型卡在零 rate」的證據。後驗責任集中在少數成分是混合模型的常見現象，不能單憑此就判定成分失效。
- HINT 用 bootstrap 重排樣本再 BottomUp 加總，無法據此證明所有跨序列依賴都建模正確；此限制對本次兩版相同。
- 未提供先前效果差的 PMM notebook、當時版本或紀錄，因此無法斷定先前是哪一個原因。只使用單一**訓練** seed；額外 5 個 seed 僅重抽固定模型的預測樣本，不能代替多次訓練或統計顯著性結論。

## 重新執行

在編輯器選擇 `.venv/bin/python` 作為 notebook 核心。上方實驗定義是實際執行的 `experiment_code.py` 原碼副本；下方留存教學片段為參考，不會在 Run All 時啟動第二套模型或 AutoARIMA。預設只讀取已記錄結果，避免重跑覆蓋。若要重訓，先另備份 `experiment_results`，再用 `run_experiment(steps=2000, budget_seconds=3600, phase='formal', evaluate_test=True)`；先驗證用 `evaluate_test=False`。

不必重新訓練即可使用已保存權重：`predict_saved_model(run_id, history, S)`；函式會重建相同設定、核對階層順序，並預測下一個 12 個月。選定 run_id 記錄於 `selection.json` 的 `run_ids`。Notebook 內附範例。三組 PMM 的訓練配置已用完本次搜尋上限，沒有另外試第四組。

原樣重跑器：`original_gmm_attempt.py`；驗證順序隔離實驗：`isolate_ordering.py`；參數診斷：`inspect_rates.py`；檢查：`verify_results.py`。原樣重跑器需設定與本次相同的 OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1、KMP_DUPLICATE_LIB_OK=TRUE。
'''
(root/'PMM實驗結果.md').write_text(report)

notebook_path=root/'ＰＰＰ.ipynb'
n=nbformat.read(notebook_path,as_version=4)
n.nbformat_minor=5
for i,cell in enumerate(n.cells):
    cell.setdefault('id', f'pmm-cell-{i:03d}')
for cell in n.cells:
    if 'experiment_definitions' in cell.metadata.get('tags',[]):
        cell.source=(root/'experiment_code.py').read_text()
    if 'experiment_run' in cell.metadata.get('tags',[]):
        cell.source='''# 預設讀取已完成的本次實驗，不會重新訓練。
RUN_EXPERIMENT = False
if RUN_EXPERIMENT:
    runs = run_experiment(steps=2000, budget_seconds=3600, phase="formal", evaluate_test=True)
else:
    display(pd.read_csv(OUT / "comparison.csv"))
    print("驗證集選擇：", json.loads((OUT / "selection.json").read_text()) if (OUT / "selection.json").exists() else "尚未完成")
'''
n.cells=[c for c in n.cells if 'experiment_report' not in c.metadata.get('tags',[])]
n.cells=[c for c in n.cells if 'saved_prediction_example' not in c.metadata.get('tags',[])]
n.cells.insert(1,nbformat.v4.new_markdown_cell(report,metadata={'tags':['experiment_report']}))
example='''# 讀取驗證集選出的設定；預設不重新訓練，也不重新計分測試集。
selection = json.loads((OUT / "selection.json").read_text())
selected_run_id = selection.get("run_ids", {}).get(selection["selected"], selection["selected"] + "_formal")
selected_config = json.loads((OUT / f"{selected_run_id}.json").read_text())
print("已選定模型：", selected_run_id)
print({key: selected_config.get(key) for key in ["loss", "lr", "weighted", "actual_steps", "validation_scrps", "test_scrps"]})
RUN_SAVED_PREDICTION = False
if RUN_SAVED_PREDICTION:
    Y, S, tags, train, validation, pretest, test = load_data()
    saved_forecast = predict_saved_model(selected_run_id, pretest, S)
    display(saved_forecast.head())
'''
n.cells.insert(4,nbformat.v4.new_code_cell(example,metadata={'tags':['saved_prediction_example']}))
n.metadata.kernelspec={'name':'pmm-experiment','display_name':'PMM experiment (.venv Python 3.10)','language':'python'}
n.metadata['experiment']={'original_sha256':session['original_sha256'],'implementation_sha256':hashlib.sha256((root/'experiment_code.py').read_bytes()).hexdigest(),'report_generated':now}
nbformat.validate(n)
nbformat.write(n,notebook_path)
assert hashlib.sha256(Path(session['original_file']).read_bytes()).hexdigest()==session['original_sha256']
print(table,flush=True)
