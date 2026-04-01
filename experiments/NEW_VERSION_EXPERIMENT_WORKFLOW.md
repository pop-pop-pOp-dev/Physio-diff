# 新版本实验流程总纲（论文对齐版）

本文档用于统一新版本 Physio-Diff（语言-机理融合版）的实验执行顺序、命令入口、产物位置与论文映射关系。  
默认在项目根目录 `Physio-Diff` 下执行命令。

---

## 1. 目标与原则

- 目标：形成“**可复现 + 可统计检验 + 可直接写论文**”的一整套证据链。
- 核心证据链：  
  `主方法（full-stack）` -> `多基线比较` -> `LLM/机理消融` -> `跨数据集迁移矩阵` -> `全局统计闭环`。
- 统计统一口径：中位数 + IQR + bootstrap CI + paired Wilcoxon + rank-biserial + BH-FDR。
- 结果落盘统一：每个 seed 独立目录，避免手工覆盖。

---

## 2. 实验入口总览（按职责）

### 2.1 训练/生成/评估主入口

- `src/scripts/run_pipeline.py`
  - 单次运行主入口（训练或复用 checkpoint、生成合成样本、TSTR、鲁棒性、语言指标）。
  - 主要产物：`physio_results.json`、`synthetic_normalized.npz`、可视化图。

### 2.2 多 seed SOTA 对比

- `experiments/sota_runs/run_multi_seed.py`
  - 同时跑 `physio_diff` 系列与全部基线（TimeGAN/CSDI/TS-Diff/cGAN/WGAN-GP/DDPM/TSGM）。
- `experiments/reports/build_multiseed_report.py`
  - 汇总多 seed 报告，生成 p/q 值和效应量。

### 2.3 LLM 全栈消融

- `experiments/ablations/run_llm_ablation_matrix.py`
  - 跑 full-stack、mechanistic-only、language-only、-cycle、-semantic、-artifact 等变体。
- `experiments/reports/build_llm_ablation_report.py`
  - 汇总消融表格（含语言指标列）。

### 2.4 多数据集协议与统计闭环

- `experiments/protocols/run_protocol_suite.py`
  - 数据集 × 方法 × fold × seed 的协议化批量执行。
- `experiments/protocols/run_cross_dataset_matrix.py`
  - 源域 -> 目标域迁移矩阵评估。
- `experiments/protocols/run_evidence_closure_suite.py`
  - 一键串联 protocol + closure 报告（建议论文主流程使用）。
- `experiments/reports/build_multi_dataset_closure_report.py`
  - 全局/局部 FDR 统计闭环报告。
- `experiments/reports/build_cross_dataset_matrix_report.py`
  - 跨数据集矩阵报告（含 Wilcoxon + effect size + FDR）。

---

## 3. 配置层（你需要固定的关键参数）

建议主配置：`configs/best_improved.yaml`

重点字段（写论文必须说明）：

- 数据与切分：
  - `data.dataset_name`
  - `data.split_strategy`（`subject` / `loso` / `groupkfold`）
  - `data.cv_n_splits`、`data.cv_fold_index`
  - `data.channels`、`target_fs`、`window_length`、`window_stride`
  - `clip_mode`、`clip_value`
- 模型：
  - `model.model_type`（`mechanistic` 或 `standard`）
- 文本/LLM 条件：
  - `text.encoder_type`（`local` / `minilm` / `bge` / `hf`）
  - 文本分支在论文中应表述为“language-grounded conditioning encoder”，不是后处理解释器
- 训练开关（LLM 融合点）：
  - `train.use_language_conditioning`
  - `train.use_text_prototypes`
  - `train.use_signal_text_cycle`
  - `train.use_semantic_alignment`
  - `train.use_artifact_text_conditioning`
- 损失权重：
  - `loss.w_mech`
  - `loss.w_text_proto`
  - `loss.w_cycle`
  - `loss.w_artifact_text`
  - `loss.w_semantic_align`
- 评估公平性参数：
  - `eval.synth_samples_per_class`
  - `eval.cfg_scale`
  - `eval.sample_steps`
  - `eval.match_stats`
  - `eval.clip_min/clip_max`
- 文本编码器（已接入主流预训练选项）：
  - `text.encoder_type`: `minilm` / `bge` / `local`
  - `text.minilm_model_name_or_path`: `all-MiniLM-L6-v2` 本地目录或 HF id
  - `text.bge_model_name_or_path`: `bge-small-en-v1.5` 本地目录或 HF id
  - `text.trainable_pretrained`: 是否微调预训练文本编码器

---

## 4. 推荐执行顺序（论文生产线）

## 4.0 一周冲刺版（单源域训练 + 多目标域仅测试）

当资源受限且希望尽量不牺牲统计严谨性时，推荐使用：

```bash
python experiments/protocols/run_source_train_target_eval.py \
  --source_config configs/source_wesad.yaml \
  --source_name wesad \
  --target_configs configs/stress_predict.yaml,configs/swell_kw.yaml,configs/case.yaml \
  --methods physio_diff,timegan,csdi,tsdiff,ddpm \
  --seeds 0,1,2,3,4 \
  --device cuda
```

该流程会：

- 在**统一源域**上完成多方法 5-seed 训练；
- 在多目标域进行**仅测试**泛化评估（不重复训练）；
- 产出源域多 seed 报告与目标域统计闭环报告（含 p/q 与效应量）。

## Step A: 单配置冒烟（先确认链路可跑）

```bash
python src/scripts/run_pipeline.py --config configs/best_improved.yaml
```

检查：

- 是否生成 `physio_results.json`
- 是否包含：
  - `tstr`
  - `robust`
  - `time_freq`
  - `language_metrics`

---

## Step B: SOTA 多 seed 主结果

```bash
python experiments/sota_runs/run_multi_seed.py \
  --seeds 0,1,2,3,4 \
  --methods physio_diff_main,physio_diff_legacy_fullstack,physio_diff_main_additive_cond,physio_diff_main_no_multiband,timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm \
  --output_root experiments/sota_runs/outputs
```

生成主报告：

```bash
python experiments/reports/build_multiseed_report.py \
  --root experiments/sota_runs/outputs \
  --primary_method physio_diff_main \
  --out_path experiments/reports/multiseed_report.md
```

---

## Step C: 架构主线消融矩阵

```bash
python experiments/ablations/run_llm_ablation_matrix.py \
  --config configs/main_competitive.yaml \
  --variants main_competitive,legacy_fullstack,additive_conditioning,no_multiband,cross_attention,with_light_freq_prior,mechanistic_only,language_only \
  --seeds 0,1,2,3,4 \
  --output_root experiments/ablations/llm_outputs
```

汇总：

```bash
python experiments/reports/build_llm_ablation_report.py \
  --root experiments/ablations/llm_outputs \
  --out_path experiments/reports/llm_ablation_report.md
```

---

## Step D: 多数据集协议 + 跨数据集 + 统计闭环（推荐一键）

```bash
python experiments/protocols/run_evidence_closure_suite.py \
  --configs configs/stress_predict.yaml,configs/swell_kw.yaml,configs/case.yaml \
  --methods physio_diff_main,physio_diff_legacy_fullstack,physio_diff_main_additive_cond,physio_diff_main_no_multiband,timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm \
  --seeds 0,1,2,3,4 \
  --split_strategy groupkfold \
  --cv_n_splits 5 \
  --output_root experiments/protocols/outputs \
  --run_cross_matrix
```

> 注：超大数据集（单数据集 >50G）可按资源策略暂不纳入主流程，先保证中等体量数据集把统计闭环跑通。

---

## 5. 输出目录规范（论文写作直接引用）

- SOTA 主对比：
  - `experiments/sota_runs/outputs/<method>/seed_<k>/...`
- LLM 消融：
  - `experiments/ablations/llm_outputs/<variant>/seed_<k>/...`
- 多数据集 protocol：
  - `experiments/protocols/outputs/<dataset>/<split>/fold_<i>/<method>/seed_<k>/...`
- 报告文件：
  - `experiments/reports/multiseed_report.md`
  - `experiments/reports/llm_ablation_report.md`
  - `experiments/reports/multi_dataset_closure_report.md`
  - `experiments/reports/cross_dataset_matrix_report.md`
  - `experiments/reports/reviewer_risk_crosswalk.md`

---

## 6. 论文章节映射（建议）

- 方法主干（模型结构 + 损失）：
  - `src/models/*`
  - `src/losses/*`
  - `src/train/train_diffusion.py`
  - `src/text/*`
- 主结果表（SOTA）：
  - `multiseed_report.md`
- 消融表（机制/语言模块）：
  - `llm_ablation_report.md`
- 泛化与迁移（跨域）：
  - `cross_dataset_matrix_report.md`
- 统计显著性闭环（审稿人关注点）：
  - `multi_dataset_closure_report.md`

---

## 7. 复现实验最小清单（投稿前必须逐项勾选）

- 固定随机种子集合（至少 5 seeds）。
- 固定 split 协议（`groupkfold` 或 `loso`）并在全文一致。
- 所有方法统一评估参数（样本数、分类器 epochs/lr、后处理范围）。
- 报告中显式给出：median、IQR、95%CI、p 值、q 值、效应量。
- 所有表格都能回溯到具体 `results.json/physio_results.json` 文件。

---

## 8. 常用附加脚本（出图/延迟/敏感性）

- 图表导出：
  - `experiments/reports/export_figures_real_pretty.py`
  - `experiments/reports/make_ablation_compare_figure.py`
  - `experiments/reports/make_fancy_plots.py`
- 延迟与资源：
  - `experiments/reports/benchmark_latency.py`
  - `experiments/reports/build_latency_table.py`
- 灵敏度分析：
  - `experiments/reports/sensitivity_eval.py`

---

## 9. 建议的“最终论文主流程”一句话版本

先跑 `run_evidence_closure_suite.py` 形成跨数据集统计闭环，再补 `run_llm_ablation_matrix.py` 强化创新解释，最后用 `build_multiseed_report.py` 与图表脚本固化主结果与可视化。

---

## 10. 投稿前一致性检查（建议固定）

以下检查项用于保证论文叙述与实验产物一致，避免出现“文稿结论先于证据文件”的风险。

### 10.1 结果文件一致性

- 主方法每个 seed 均应存在：
  - `physio_results.json`
- 基线每个 seed 均应存在：
  - `results.json`
- 所有报告脚本应基于同一批输出目录重建，避免混用旧版本结果。

### 10.2 统计口径一致性

- 必须统一使用：
  - median (Q1--Q3)
  - bootstrap 95\% CI
  - paired Wilcoxon
  - rank-biserial effect size
  - BH-FDR 校正（q-value）
- 不允许只报告点估计而省略不显著结果解释。

### 10.3 评估公平性一致性

- 所有方法使用同一 TSTR 分类器配置与训练预算。
- 所有方法使用同一后处理规则（`match_stats`、clip 范围等）。
- 鲁棒性比较使用同一 corruption 列表与 severity 配置。

### 10.4 方法叙述一致性

- 主文默认表述：
  - 标签语义稳定性来自训练期锚定（class/prototype/text）
  - 验证期 label flip 仅作诊断，不作为核心方法
- 主文必须显式区分：
  - primary evidence（multi-seed / protocolized）
  - supplementary diagnostics（legacy/single-run）

### 10.5 计算成本一致性

- 延迟表必须由 `benchmark_latency.py` 和 `build_latency_table.py` 产出。
- 论文中关于采样开销与加速方向（latent diffusion / distillation / consistency）应与脚本输出一致。
