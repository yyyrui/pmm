# GMM → PMM 實驗紀錄

產生時間：2026-10-29T13:32:49+08:00。本次總時限：2026-10-29 12:33:57 至 13:33:57（台北時間，包含安裝、環境排錯與中斷等待）。

原檔 `hierarchical_forecasting.ipynb` 未修改；修改版為 `ＰＰＰ.ipynb`（全形英文字母）。原檔 SHA256：`074872c915b210f84833c8eed11e238856e01ac294d90003d68fc15cf66a6671`。

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

原樣 GMM：狀態 `completed`，實際 2000 步，耗時 151.8 秒。其驗證 pooled sCRPS = 3.548462。另固定同一組原始權重、不重訓，只修正推論順序，驗證 pooled sCRPS = 0.116832（seed=1）。

共同設定：h=12、input_size=24、10 components、robust scaler、month 歷史特徵、seed=1、1,000 samples、99 個訓練／驗證分位數、原預設 NHITS 架構與 batch/window 設定。保留與原程式相同的最後訓練權重策略；原程式未儲存／還原驗證最佳 checkpoint。

| PMM 設定 | 學習率 | mixture weights | 動機 |
|---|---:|---|---|
| A：直接替換 | 0.001 | 固定等權重 | 測試只換損失的效果 |
| B | 0.0003 | 固定等權重 | A 短跑偏高估，檢查較小更新是否穩定 |
| C | 0.0003 | 可學習權重 | 與 B 比較，檢查固定每成分 10% 機率是否過度受限 |

| 版本 | 階段 | 實際步數 | 狀態 | 驗證 pooled sCRPS | 測試 pooled sCRPS |
|---|---|---:|---|---:|---:|
| GMM_fixed | short | 200 | completed | 0.178488 | — |
| PMM_A | short | 200 | completed | 0.395771 | — |
| PMM_B | short | 50 | interrupted | — | — |
| GMM_fixed | formal | 2000 | completed | 0.120999 | 0.117362 |
| PMM_A | formal | 1550 | completed | 0.399246 | 0.409650 |
| PMM_B | formal | 1250 | interrupted | — | — |


`short` 僅為試跑；`formal` 表示要求原 2,000 步預算或原早停規則，仍須看實際步數與狀態。`time_limited` 是因時限中止，不能當作完成完整預算，也不會用來與完成組公平選參數。沒有測試分數的 PMM 設定代表未被選中或未完成，並非分數遺失。

目前選定：`PMM_A`；選擇依據：`validation Overall pooled_scaled_crps among completed PMM runs only`。最終測試是否完成：`True`。

未完成／排除紀錄：`{'PMM_B': 'interrupted; no complete validation result', 'PMM_C': 'not run within time budget'}`。若僅 A 完成，這表示只有一組可用的 PMM 完整結果，不能宣稱已成功完成三組調參。C 的表格為原定設定，未執行時不得視為實驗結果。原樣 GMM 最後測試 pooled sCRPS：3.670745，其排序錯誤仍保留，僅作問題重現。

## 指標與數值檢查

主要比較值採 `pooled_scaled_crps = 2 × Σ pinball(y, q) / (98 × Σ |y|)`，加總範圍為該階層所有序列與 12 個月，98 個 q 與原評估的 level=2,4,…,98 對應（不含中位數）。這是以總絕對目標作分母的尺度化 CRPS 近似；不同於目前套件先除每條序列各自分母再平均。原 notebook 的歷史 Overall=0.122793 只是舊輸出，不能拿來當本次基準，也不能假定套件版本改變前後指標實作完全相同。

`*_metrics.csv` 保留原套件 scaled_crps、MAE、RMSE；`*_pooled_metrics.csv` 存穩定的共同指標。`*_diagnostics.csv` 提供各層 90% 區間覆蓋率、寬度、偏差與負平均預測數；`*_predictions.parquet` 保留實際預測。比較前檢查列數、ID／日期、finite 值、分位數單調性與階層平均值一致性。`verification.json` 另以套件在合併 ID 後的 scaled_crps 交叉驗證手算 pooled 公式。

## 已確認的事實

- 實際資料沒有負值，但 112973 筆（89.28%）不是整數。原程式在切分後才歸零負值，對已切出的訓練資料不起作用；此資料沒有負值，所以本次不是造成差異的原因。
- PMM 程式可以接受非負小數；0.25 的 likelihood 與梯度試算均為有限值。套件把 Poisson support 放寬成 nonnegative，並以 gamma 函數延伸計算。因此「能執行」不代表這是符合資料生成過程的離散機率模型。
- 原程式的 ID 順序與 S 不符；固定同一組已訓練權重，只改推論順序即可大幅改善驗證分數，確認排序／階層對應有實質影響。
- 驗證集有全零序列，套件逐序列 scaled_crps 的 epsilon 分母會使短跑分數達 10^13 等級。原指標仍完整保留；兩版都用同一個 pooled 定義作主要比較。
- 修正後的預測平均值通過階層加總檢查；沒有把分位數誤當成可直接相加的量。
- 官方 PMM 支援 n_components、quantiles、num_samples、weighted 等參數；本次沒有修改套件原始碼。[Nixtla PMM 文件](https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html#poisson-mixture-mesh-pmm)

## 可能原因與尚未確認的部分

- 旅遊資料多為實數估計值，Poisson 的離散計數假設可能不匹配；不能單憑小數比例證明它是誤差主因。
- 每個 Poisson 成分的平均數與變異數綁定；混合後仍可以過度離散，不能簡化成「PMM 的變異數必等於平均數」。相較 GMM 的個別可學習標準差，PMM 可能需要不同的成分位置／權重才能表達尾端、零值及季節變動。
- 訓練期逐序列樣本變異數／平均數中位數約 44.66。這包含季節性與時間變化，**不是條件過度離散的證明**。
- PMM A 短跑有偏高估現象；rate 下界與混合權重診斷見 `*_rate_diagnostics.json`。softplus 下界可能影響梯度，但本次僅檢查預測參數，不能斷言發生梯度消失或成分崩潰。
- HINT 用 bootstrap 重排樣本再 BottomUp 加總，無法據此證明所有跨序列依賴都建模正確；此限制對本次兩版相同。
- 未提供先前效果差的 PMM notebook、當時版本或紀錄，因此無法斷定先前是哪一個原因。只執行單一 seed，沒有多次重複或統計顯著性結論。

## 重新執行

在編輯器選擇 `.venv/bin/python` 作為 notebook 核心。上方實驗定義是實際執行的 `experiment_code.py` 原碼副本；下方留存教學片段為參考，不會在 Run All 時啟動第二套模型或 AutoARIMA。預設只讀取已記錄結果，避免重跑覆蓋。若要重訓，先另備份 `experiment_results`，再用 `run_experiment(steps=2000, budget_seconds=3600, phase='formal', evaluate_test=True)`；先驗證用 `evaluate_test=False`。

原樣重跑器：`original_gmm_attempt.py`；驗證順序隔離實驗：`isolate_ordering.py`；參數診斷：`inspect_rates.py`；檢查：`verify_results.py`。原樣重跑器需設定與本次相同的 OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1、KMP_DUPLICATE_LIB_OK=TRUE。
