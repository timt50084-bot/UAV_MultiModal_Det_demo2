# UAV Multi-Modal OBB Detection

面向无人机场景的 RGB + IR 双模态旋转框检测与阶段式跟踪框架，覆盖数据准备、训练、验证、推理、实验编排和结果汇总。项目主线不是“最简单的 YOLOv8 封装”，而是围绕双模态输入、小目标、OBB、temporal 和记忆模块做了专门设计，并保留了 tracking 的阶段性扩展能力。

## 项目简介

这个仓库的主要目标是为 UAV / DroneVehicle 类场景提供一套可落地的训练与实验框架：

- 检测主线支持 RGB + IR 双模态输入。
- 检测框为 OBB（rotated box），不是普通水平框。
- 主线配置已经包含小目标增强、双模态融合、temporal memory、角度感知分配等能力。
- tracking 已有分阶段实现与评估配置，但建议和检测主线区分理解：检测 `full_project` 是日常主入口，tracking 更适合按阶段配置做实验和对比。

## 核心能力

- 双模态输入：`DroneDualDataset` 按配对的 RGB / IR 图像读取数据，训练和推理都走双输入接口。
- OBB 检测：从数据准备、标签格式、检测头、损失到评估都围绕旋转框实现。
- 小目标优化：主线配置启用了小目标导向采样、增强型 neck、tiny-aware assigner 和 `mAP_S` 导向的 checkpoint 选择。
- 多模态融合：主线模型使用 `ReliabilityAwareFusion`，不是简单拼接输入。
- Temporal / memory：主线检测配置默认启用 `model.temporal.enabled: true` 和 `mode: memory`，用于时序稳定检测。
- 阶段式 tracking：`configs/main/tracking_*.yaml` 提供从 base 到 final 的逐步增强版本，适合做跟踪实验和消融，不建议和日常检测入口混用。
- 工程化闭环：训练、验证、推理、实验计划、结果汇总、模板文档都在仓库里，有统一的输出目录与配置落盘机制。

## 目录结构

- `configs/`
  - 主配置、数据配置、模型配置和历史实验配置。
  - 日常训练优先看 `configs/main/full_project.yaml`。
  - `configs/main/` 下还放了检测消融配置和 tracking 阶段配置。
  - `configs/archive/` 主要用于历史实验和兼容，不是日常入口。
- `src/model/`
  - 检测模型实现，包括 backbone、neck、head、fusion、temporal、detector 组装逻辑。
- `src/engine/`
  - 训练、验证、checkpoint、回调、评估驱动逻辑。
- `src/data/`
  - 数据集、采样、增强、预处理相关实现。
- `src/tracking/`
  - 跟踪主线代码，包括 association、memory、modality、refinement、metrics、visualization。
- `tools/`
  - 日常命令入口：`train.py`、`val.py`、`infer.py`、`prepare_dronevehicle_dataset.py`、`run_experiment_suite.py`、`summarize_results.py`。
- `tests/`
  - 数据准备、配置、检测评估、tracking 各阶段和项目 smoke test。
- `docs/`
  - 实验计划、技术方案、PPT 大纲、消融表、Demo 脚本、tracking 结果模板等辅助文档。
- `scripts/`
  - 一些较早期的 shell 辅助脚本。当前日常使用优先走 `tools/` 下的 Python 入口。

## 环境安装

仓库里同时有 `requirements.txt` 和 `setup.py`。`setup.py` 的 `python_requires` 为 `>=3.8`，当前仓库更推荐使用 Python 3.10 左右的 Conda 环境。

```bash
conda create -n drone python=3.10 -y
conda activate drone
pip install -r requirements.txt
pip install -e . --no-deps
```

说明：

- `requirements.txt` 已包含当前主线依赖，README 不再额外假设不存在的部署依赖或 benchmark 环境。
- 如果你只想先检查环境是否通，建议先跑一个最小测试：

```bash
python -m unittest tests.test_experiment_configs
```

## 数据准备

### 1. 日常需要改的路径

主训练配置的数据根目录在：

- `configs/main/full_project.yaml` 的 `dataset.root_dir`

默认值通常是：

```yaml
dataset:
  root_dir: D:/DataSet/DroneVehicle_Processed
```

如果你使用自己的处理后数据目录，优先改这里。

### 2. 处理后数据组织

当前主线数据集 `DroneDualDataset` 期望处理后的目录大致如下：

```text
<root_dir>/
  train/
    images/
      img/
      imgr/
    labels/
      merged/
  val/
    images/
      img/
      imgr/
    labels/
      merged/
```

其中：

- `img/` 放 RGB 图像。
- `imgr/` 放 IR 图像。
- `merged/` 放融合后的 OBB 标签。

### 3. 数据准备脚本

仓库提供了专门的数据准备脚本：

```bash
python tools/prepare_dronevehicle_dataset.py \
  --raw-root D:/DataSet/DroneVehicle \
  --output-root D:/DataSet/DroneVehicle_Processed \
  --splits train val
```

这个脚本会基于当前代码完成：

- RGB / IR 文件配对。
- XML 到 YOLO OBB 标签转换。
- RGB / IR 标签融合后写入 `labels/merged/`。
- 生成 `DroneDualDataset` 直接可读的目录结构。

注意：

- 主流程依赖 RGB / IR 文件名 stem 对齐，双模态图像需要一一对应。
- temporal / memory 检测和部分 tracking 逻辑会按序列顺序读取相邻帧，目录内文件命名和排序应保持稳定。
- 如果你不确定自己的原始数据是否符合脚本假设，先看 `tools/prepare_dronevehicle_dataset.py` 和 `src/data/datasets/drone_rgb_ir.py`，不要直接猜目录格式。

## 配置系统说明

这一部分是当前仓库最重要的入口说明。

### 1. 日常训练优先改哪个文件

日常 full project 训练，优先修改：

- `configs/main/full_project.yaml`

这是当前主线推荐入口。对于第一次接触仓库的人，不建议从 `configs/archive/` 或旧的 `configs/exp_*.yaml` 开始。

### 2. 其他配置文件分别是什么

- `configs/main/baseline.yaml`
  - 基础检测配置，用于对照实验。
- `configs/main/fusion_main.yaml`
  - 以融合模块为重点的检测实验配置。
- `configs/main/assigner_main.yaml`
  - 以 assigner / 小目标匹配策略为重点的检测实验配置。
- `configs/main/temporal_main.yaml`
  - 以 temporal memory 检测为重点的检测实验配置。
- `configs/main/tracking_base.yaml` 到 `configs/main/tracking_final.yaml`
  - 跟踪阶段配置，按功能逐步增强，不是日常检测训练的统一主入口。
- `configs/main/tracking_eval.yaml`
  - 离线 tracking 评估入口。
- `configs/archive/`
  - 历史实验和兼容配置。可以复现旧实验，但不建议作为日常入口。

### 3. 现在常用参数建议改哪里

高频参数建议直接在 `configs/main/full_project.yaml` 中修改：

- `dataloader.batch_size`
- `dataloader.num_workers`
- `dataset.imgsz`
- `train.epochs`
- `train.lr`
- `train.accumulate`
- `performance.profile_train`

当前这轮整理后，高常用 loader 和训练参数的规范入口已经统一到了主配置，不需要再去翻多层历史实验配置。

如果你还需要进一步调数据加载性能，也优先看：

- `performance.dataloader.persistent_workers`
- `performance.dataloader.prefetch_factor`

### 4. effective config 与 resolved_config.yaml

训练启动时，`tools/train.py` 会做两件事：

- 在控制台打印一份 effective config 摘要，帮助你确认当前真正生效的配置。
- 将最终合并后的配置保存为：

```text
outputs/experiments/<run_name>/resolved_config.yaml
```

这份 `resolved_config.yaml` 会包含默认配置、主配置、兼容重定向和命令行覆盖后的最终结果，是排查“到底用了什么参数”的第一入口。

### 5. 旧字段兼容与 warning

当前配置加载逻辑保留了旧字段兼容和旧路径重定向能力：

- 旧的 `configs/exp_*.yaml` 路径仍可通过重定向加载。
- 如果旧字段和新字段同时存在，配置系统会发出 warning。

建议：

- 新实验直接使用 `configs/main/*.yaml`。
- 看到 warning 时，以 `resolved_config.yaml` 为准确认最终生效字段。
- 部分实验编排脚本输出仍可能显示旧的 `configs/exp_*.yaml` 兼容路径，这是重定向层在工作；手动运行和日常修改时仍建议优先使用 `configs/main/*.yaml`。

如需更细的字段说明，可以继续参考 [`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md)，但日常上手以本 README 为主即可。

## 常用命令

### 1. 主线训练

```bash
python tools/train.py --config configs/main/full_project.yaml --device 0
```

### 2. 只跑 1 个 epoch 快速试跑

先把 `configs/main/full_project.yaml` 中的 `train.epochs` 改成 `1`，然后执行：

```bash
python tools/train.py --config configs/main/full_project.yaml --device 0
```

如果你只想验证配置链路，也可以先把 `dataloader.batch_size` 调小、`dataset.imgsz` 调低后再跑。

### 3. 验证

```bash
python tools/val.py \
  --config configs/main/full_project.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --device 0
```

### 4. 推理

单张图或成对图像推理：

```bash
python tools/infer.py \
  --config configs/main/full_project.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --source_rgb path/to/rgb.jpg \
  --source_ir path/to/ir.jpg \
  --save_dir outputs/infer_full_project
```

序列目录推理：

```bash
python tools/infer.py \
  --config configs/main/full_project.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --source_rgb path/to/rgb_sequence \
  --source_ir path/to/ir_sequence \
  --save_dir outputs/infer_sequence
```

### 5. tracking 阶段配置示例

如果你要跑跟踪阶段配置，请显式切换到对应的 tracking 配置，例如：

```bash
python tools/infer.py \
  --config configs/main/tracking_final.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --source_rgb path/to/rgb_sequence \
  --source_ir path/to/ir_sequence \
  --save_dir outputs/tracking_final
```

说明：

- tracking 相关配置建议按阶段理解和使用。
- `tracking_final.yaml` 属于较完整的阶段配置，不代表所有 tracking 配置都要日常一起改。

### 6. 实验编排与汇总

查看实验套件计划：

```bash
python tools/run_experiment_suite.py --mode plan --subset all
```

汇总已有结果：

```bash
python tools/summarize_results.py --experiments-root outputs/experiments
```

## 训练输出说明

当 `experiment.enable_unified_dirs: true` 时，项目会把输出整理到统一的实验目录下。以 `full_project` 为例，常见输出如下：

```text
outputs/
  experiments/
    full_project/
      resolved_config.yaml
      weights/
        latest.pt
        best.pt
      error_analysis/
      tracking_eval/
```

重点说明：

- 权重默认保存到 `outputs/experiments/<run_name>/weights/`
- 最值得先看的文件是 `resolved_config.yaml`
- `best.pt` 由当前 checkpoint 逻辑自动选择，`latest.pt` 会持续更新
- 如果启用了错误分析或 tracking 评估，输出也会统一落到当前实验目录下

训练日志里优先关注：

- 启动时打印的 effective config 摘要
- 每个 epoch 的训练损失和验证指标
- `mAP_50`、`mAP_50_95`、`mAP_S`
- 配置兼容 warning 或数据路径 warning

## 性能与调参建议

- `dataloader.batch_size`、`dataset.imgsz`、`dataloader.num_workers` 会直接影响显存占用、吞吐和首轮加载速度。
- `train.accumulate` 可以在显存不足时帮助维持等效 batch，但训练时间通常会增加。
- `performance.profile_train` 是诊断开关，不是提速开关。它用于看数据加载和训练阶段耗时，排查瓶颈时再开。
- `performance.dataloader.persistent_workers` 和 `prefetch_factor` 主要在 `num_workers > 0` 时有意义。

如果你只是想先快速试跑一遍，推荐：

- 把 `train.epochs` 改成 `1`
- 把 `dataloader.batch_size` 调到显存能稳定承受的值
- 先用 `dataset.imgsz: 640` 或更低分辨率
- 必要时暂时把 `dataloader.num_workers` 调低，先确认环境和数据链路没问题

## 当前状态与注意事项

- 当前主线推荐入口是 `configs/main/full_project.yaml`。
- `configs/main/` 下的其他检测配置主要用于模块消融和专项实验。
- `configs/main/tracking_*.yaml` 是阶段式 tracking 配置，不要把它们理解成日常都要同时修改的一组主配置。
- `configs/archive/` 和旧 `configs/exp_*.yaml` 主要用于兼容和历史实验复现，不建议新用户优先使用。
- `tools/infer.py` 是当前推荐的推理入口；`tools/track.py` 更接近兼容或较早期的跟踪封装。
- `docs/` 下的模板文档适合整理实验、答辩或交付材料，但不代替训练入口说明。

## 补充文档

- 配置说明：[`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md)
- 实验计划模板：[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
- 技术方案模板：[`docs/TECHNICAL_PLAN_TEMPLATE.md`](docs/TECHNICAL_PLAN_TEMPLATE.md)
- PPT 大纲：[`docs/PPT_OUTLINE.md`](docs/PPT_OUTLINE.md)
- 消融表模板：[`docs/ABLATION_TABLE_TEMPLATE.md`](docs/ABLATION_TABLE_TEMPLATE.md)
- Tracking 结果模板：[`docs/RESULTS_TRACKING_TEMPLATE.md`](docs/RESULTS_TRACKING_TEMPLATE.md)
