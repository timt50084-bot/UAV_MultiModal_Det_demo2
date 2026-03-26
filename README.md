# UAV Multi-Modal OBB Detection

面向全天候 UAV 交通场景的 RGB-T 小目标旋转框检测系统。项目当前以“统一配置入口 + 清晰主创新主线 + 可扩展验证体系”为主线，适合作为后续系统训练、消融实验、比赛展示和技术文档整理的工程基线。

## Project Focus

当前推荐把项目收敛为 3 条主创新主线：

1. 可信度感知的多模态融合
2. 面向极小旋转目标的动态分配与检测
3. 全天候视频场景下的时序稳定检测

这 3 条主线分别对应独立实验配置，也能组合成统一的 `full_project` 主版本。

## Recommended Entry Configs

推荐优先使用以下 5 个配置作为统一实验入口：

- `configs/exp_baseline.yaml`
- `configs/exp_fusion_main.yaml`
- `configs/exp_assigner_main.yaml`
- `configs/exp_temporal_main.yaml`
- `configs/exp_full_project.yaml`

旧配置依然兼容，例如：

- `configs/exp_reliability_fusion.yaml`
- `configs/exp_angle_aware_assigner.yaml`
- `configs/exp_temporal_memory.yaml`
- `configs/exp_task_metrics.yaml`
- `configs/exp_eval_full.yaml`
- `configs/exp_error_analysis.yaml`

推荐新入口，旧配置不删除。

## Main Innovation Mapping

### 1. 可信度感知的多模态融合

- 目标：提升 RGB / IR 在夜间、天气扰动和局部退化场景下的互补建模能力
- 代码位置：
  - `src/model/fusion/reliability_fusion.py`
  - `src/model/fusion/rdm_fusion.py`
  - `src/model/fusion/`
- 推荐配置：
  - `configs/exp_fusion_main.yaml`
- 主要指标：
  - `mAP_50`
  - `GroupedMetrics` 中的 `time_of_day / weather`
  - `RGBDrop / IRDrop` 鲁棒性指标
- 说明：
  - `ReliabilityAwareFusion` 是当前推荐主线
  - `RDMFusion` 保留为可选实验分支

### 2. 面向极小旋转目标的动态分配与检测

- 目标：提升极小、细长、角度敏感目标的召回与定位稳定性
- 代码位置：
  - `src/loss/assigners/target_assigner.py`
  - `src/model/heads/obb_decoupled_head.py`
  - `src/model/necks/enhanced_neck.py`
- 推荐配置：
  - `configs/exp_assigner_main.yaml`
- 主要指标：
  - `mAP_S`
  - `Recall_S`
  - error analysis 中的 `tiny / elongated / confusion` 统计
- 说明：
  - 当前主线以 improved `DynamicTinyOBBAssigner` 的增强开关表达
  - baseline 中通过关闭 angle-aware / tiny boost 等增强项实现兼容对照

### 3. 全天候视频场景下的时序稳定检测

- 目标：降低远距离小目标在视频序列中的闪烁与不稳定检测
- 代码位置：
  - `src/model/temporal/temporal_fpn.py`
  - `src/model/temporal/temporal_memory.py`
  - `src/model/temporal/`
- 推荐配置：
  - `configs/exp_temporal_main.yaml`
- 主要指标：
  - `TemporalStability`
  - 夜间 / 低纹理场景的 grouped analysis
  - 错误分析中的逐图漏检记录
- 说明：
  - 当前推荐主线是 `temporal.mode=memory`
  - `two_frame` 仍保留为兼容方案

## Recommended Experiment Order

推荐按以下顺序推进核心实验：

1. `baseline`
2. `fusion_main`
3. `assigner_main`
4. `temporal_main`
5. `full_project`

这样安排的原因：

- `baseline` 用来固定统一对照组
- `fusion_main` 先验证全天候模态互补收益
- `assigner_main` 再验证极小目标主线收益
- `temporal_main` 单独确认视频稳定性收益
- `full_project` 最后验证三条主线是否可叠加

更细的实验规划见：

- `docs/EXPERIMENT_PLAN.md`

## Installation

```bash
git clone https://github.com/timt50084-bot/UAV_MultiModal_Det_demo2.git
cd UAV_MultiModal_Det_demo2
pip install -r requirements.txt
pip install -e .
```

## Runtime Naming Convention

新推荐配置统一使用：

- `experiment.name`
- `experiment.output_root`
- `experiment.enable_unified_dirs`

当 `experiment.enable_unified_dirs=True` 时：

- 训练权重目录：`outputs/experiments/<run_name>/weights`
- 错误分析目录：`outputs/experiments/<run_name>/error_analysis`

当前推荐命名：

- `baseline`
- `fusion_main`
- `assigner_main`
- `temporal_main`
- `full_project`

## Training

```bash
python tools/train.py --config configs/exp_baseline.yaml
python tools/train.py --config configs/exp_fusion_main.yaml
python tools/train.py --config configs/exp_assigner_main.yaml
python tools/train.py --config configs/exp_temporal_main.yaml
python tools/train.py --config configs/exp_full_project.yaml
```

## Validation

```bash
python tools/val.py --config configs/exp_full_project.yaml --weights outputs/experiments/full_project/weights/best.pt
```

若需要更完整的验证或错误分析，可在验证时叠加 CLI 覆盖：

```bash
python tools/val.py --config configs/exp_full_project.yaml --weights outputs/experiments/full_project/weights/best.pt eval.group_eval.enabled=True eval.error_analysis.enabled=True
```

也可以直接使用：

- `configs/exp_eval_full.yaml`
- `configs/exp_error_analysis.yaml`

## Validation and Analysis

当前项目支持以下验证与分析能力：

- 总体检测指标：`mAP_50 / mAP_50_95 / Precision / Recall`
- 小目标指标：`mAP_S / Recall_S / Precision_S`
- 多模态鲁棒性：`RGBDrop / IRDrop / CrossModalRobustness_*`
- 时序稳定性：`TemporalStability`
- 场景分层评估：`GroupedMetrics`
- 错误分析摘要：`ErrorAnalysis`

兼容性说明：

- 缺少 `sequence_id / frame_index / video_id` 时，`TemporalStability` 会优雅跳过
- 缺少 `day/night/weather/occlusion/size` metadata 时，`GroupedMetrics` 会返回空结果
- 缺少 drop 结果时，模态贡献分析会自动退化

## Minimal Test Commands

```bash
python -m unittest tests.test_experiment_configs
python -m unittest tests.test_eval_full
python -m unittest tests.test_error_analysis
```

## Repository Structure

```text
configs/                 experiment configs
docs/                    lightweight experiment docs
scripts/                 optional shell helpers
src/model/fusion/        fusion mainline modules
src/loss/assigners/      assigner mainline modules
src/model/temporal/      temporal mainline modules
src/metrics/             evaluation and error analysis
tools/train.py           training entry
tools/val.py             validation entry
```

## Notes

- 当前阶段重点是统一工程入口和主线叙事，不夸大尚未完全验证的能力
- 建议后续所有系统训练与展示，优先围绕上述 5 个核心配置展开
