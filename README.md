# UAV Multi-Modal OBB Detection

面向无人机场景的 RGB + IR 双模态旋转框检测与阶段式跟踪框架，覆盖数据准备、训练、验证、推理、实验编排和结果汇总。当前检测主线明确收敛为 RGB+IR 双流输入、OBB 检测、`ReliabilityAwareFusion` 和双帧 `two_frame` temporal；tracking 主线收敛为增强版 tracking-by-detection (`tracking_final`)。

## 项目简介

这个仓库的主要目标是为 UAV / DroneVehicle 类场景提供一套可落地的训练与实验框架：

- 检测主线支持 RGB + IR 双模态输入。
- 检测框为 OBB（rotated box），不是普通水平框。
- 主线配置已经包含小目标增强、`ReliabilityAwareFusion`、双帧 temporal、角度感知分配等能力。
- tracking 主入口保留 `tracking_base` 和 `tracking_final`；`tracking_eval` 仅作为离线评估辅助入口；中间阶段配置已降级为历史实验入口并移到 `configs/archive/tracking/`。

## 核心能力

- 双模态输入：`DroneDualDataset` 按配对的 RGB / IR 图像读取数据，训练和推理都走双输入接口。
- OBB 检测：从数据准备、标签格式、检测头、损失到评估都围绕旋转框实现。
- 小目标优化：主线配置启用了小目标导向采样、增强型 neck、tiny-aware assigner 和 `mAP_S` 导向的 checkpoint 选择。
- 多模态融合：主线模型使用 `ReliabilityAwareFusion`，baseline 对照保留 `SimpleConcatFusion`。
- Temporal：主线检测配置默认启用 `model.temporal.enabled: true` 和 `mode: two_frame`，用于双帧时序稳定检测。
- Tracking：主入口保留 `configs/main/tracking_base.yaml` 和 `configs/main/tracking_final.yaml`；`configs/main/tracking_eval.yaml` 仅用于离线评估；归档阶段配置仅用于历史实验或兼容。
- 工程化闭环：训练、验证、推理、实验计划、结果汇总、模板文档都在仓库里，有统一的输出目录与配置落盘机制。

## 目录结构

- `configs/`
  - 主配置、数据配置、模型配置和历史实验配置。
  - 日常训练优先看 `configs/main/full_project.yaml`。
  - `configs/main/` 下保留检测消融配置，以及 tracking 的主入口 `tracking_base.yaml` / `tracking_final.yaml`；`tracking_eval.yaml` 作为离线评估辅助入口保留。
  - `configs/archive/` 主要用于历史实验和兼容，不是日常入口；已归档的 tracking 中间阶段配置也放在这里。
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
  root_dir: dataset/DroneVehicle_process
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
  --raw-root dataset/DroneVehicle \
  --output-root dataset/DroneVehicle_process \
  --splits train val
```

这个脚本会基于当前代码完成：

- RGB / IR 文件配对。
- XML 到 YOLO OBB 标签转换。
- RGB / IR 标签融合后写入 `labels/merged/`。
- 生成 `DroneDualDataset` 直接可读的目录结构。

注意：

- 主流程依赖 RGB / IR 文件名 stem 对齐，双模态图像需要一一对应。
- 双帧 temporal 检测和 `tracking_final` 中的 tracking memory 逻辑会按序列顺序读取相邻帧，目录内文件命名和排序应保持稳定。
- `dataset.root_dir` 主线保持为 `dataset/DroneVehicle_process`；训练 / 验证时会按仓库根目录归一化这个相对路径，因此不从 repo 根目录启动也不会直接丢失主线数据目录。
- 如果你不确定自己的原始数据是否符合脚本假设，先看 `tools/prepare_dronevehicle_dataset.py` 和 `src/data/datasets/drone_rgb_ir.py`，不要直接猜目录格式。

## 配置系统说明

这一部分是当前仓库最重要的入口说明。

### 1. 日常训练优先改哪个文件

日常 full project 训练，优先修改：

- `configs/main/full_project.yaml`

这是当前主线推荐入口。对于第一次接触仓库的人，不建议从 `configs/archive/` 或旧的 `configs/exp_*.yaml` 开始。

### 2. 其他配置文件分别是什么

- `configs/main/baseline.yaml`
  - 基础检测配置，用于“简单融合 + 无时序增强”的最小对照实验。
- `configs/main/fusion_main.yaml`
  - 以融合模块为重点的检测实验配置。
- `configs/main/assigner_main.yaml`
  - 以 assigner / 小目标匹配策略为重点的检测实验配置。
- `configs/main/temporal_main.yaml`
  - 以双帧 temporal 检测为重点的检测实验配置。
- `configs/main/tracking_base.yaml`
  - tracking-by-detection 基础版主入口。
- `configs/main/tracking_final.yaml`
  - 增强版 tracking-by-detection 主入口。
- `configs/main/tracking_eval.yaml`
  - 离线 tracking 评估辅助入口，不是 tracking 主线训练 / 推理入口。
- `configs/archive/tracking/`
  - 历史 tracking 中间阶段配置，仅用于旧实验复现或兼容加载。
- `configs/archive/`
  - 历史实验和兼容配置。可以复现旧实验，但不建议作为日常入口。

这些 `configs/main/*.yaml` 主线入口默认继承 `full_project.yaml` 的数据路径和训练日程，除非文件内显式覆盖。

### 3. 现在常用参数建议改哪里

高频参数建议直接在 `configs/main/full_project.yaml` 中修改：

- `dataloader.batch_size`
- `dataloader.num_workers`
- `dataset.imgsz`
- `train.epochs`
- `train.patience`
- `train.eval_interval`
- `train.lr`
- `train.accumulate`
- `train.use_amp`
- `performance.profile_train`

当前这轮整理后，高常用 loader 和训练参数的规范入口已经统一到了主配置，不需要再去翻多层历史实验配置。

当前 `full_project` 主线默认值是：

- `dataset.root_dir: dataset/DroneVehicle_process`
- `dataloader.batch_size: 4`
- `dataset.imgsz: 1024`
- `train.epochs: 300`
- `train.patience: 50`
- `train.eval_interval: 5`
- `train.lr: 0.0003`
- `train.use_amp: False`
- `eval.evaluator: gpu`
- `eval.obb_iou_backend: gpu_prob`

如果 `1024` 对当前设备压力过大，不建议偷偷改小主线默认值；优先在命令行临时覆盖 `dataset.imgsz=960` 或 `dataset.imgsz=800`。
`train.eval_interval` 控制训练期间的自动验证频率，最后一轮会强制验证一次。early stopping 的 `patience` 仍按训练 epoch 计数，只是在发生验证的 epoch 上检查是否触发。
detection 验证主线默认现在是 `eval.evaluator=gpu` + `eval.obb_iou_backend=gpu_prob`。这条路径使用 GPU evaluator 和 ProbIoU surrogate，相比 CPU shapely reference 更快，但它不是 exact polygon IoU。
CPU 不再是 detection 并列主线，只保留三类角色：显式 reference（`cpu + cpu_polygon`）、无 CUDA 时的自动 fallback，以及仍保留的 CPU analysis 路径（例如 `polygon_iou` / `ErrorAnalysis`）。
如果你需要强制使用 CPU reference，请显式覆盖：

- `eval.evaluator=cpu`
- `eval.obb_iou_backend=cpu_polygon`

如果当前环境没有 CUDA，默认验证会打印 warning 并自动回退到 CPU reference 路径，而不是直接失败。

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
- 旧的 `fusion_att_type` 仍可兼容加载，但新的配置和模型构建应统一使用 `model.fusion.type`。
- detection 侧旧的 temporal memory 路线仍可兼容加载，但它只用于历史实验或兼容；当前检测主线仍然是 `two_frame` temporal。`tracking_final` 里的 tracking memory 不受这条降级说明影响。

建议：

- 新实验直接使用 `configs/main/*.yaml`。
- 看到 warning 时，以 `resolved_config.yaml` 为准确认最终生效字段。
- 实验编排脚本默认也已经使用 `configs/main/*.yaml`；旧的 `configs/exp_*.yaml` 主要只保留给兼容调用。

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

如果你只想验证配置链路，也可以先把 `dataloader.batch_size` 调小，或临时把 `dataset.imgsz` 降到 `960` / `800` / `640` 后再跑。

### 3. 验证

```bash
python tools/val.py \
  --config configs/main/full_project.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --device 0
```

说明：

- 在 CUDA 环境下，这条命令默认走 detection GPU evaluator：`eval.evaluator=gpu` + `eval.obb_iou_backend=gpu_prob`。
- `gpu_prob` 是 ProbIoU surrogate，相比 CPU shapely reference 更快，但不是 exact polygon IoU。
- 如果当前环境没有 CUDA，运行时会打印 warning 并自动回退到 `cpu + cpu_polygon`。
- 如果你要强制使用 CPU reference，可在命令行追加：`eval.evaluator=cpu eval.obb_iou_backend=cpu_polygon`
- 这条默认切换只针对 detection 验证；tracking evaluator 仍是独立路径，没有一起切到同样的 GPU 主线。

### 3.1 detection evaluator CPU/GPU 对比

如果你要做 CPU vs GPU detection evaluator 的 parity / regression 检查，可使用：

```bash
python tools/compare_detection_evaluators.py \
  --config configs/main/full_project.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --device 0
```

这个工具固定比较：

- `cpu + cpu_polygon`
- `gpu + gpu_prob`

用途是做 detection evaluator 的 drift / runtime 对比与回归守门，不包含 tracking evaluator；输出会同时落盘结构化 JSON 和人类可读 Markdown 报告。

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

### 5. tracking 主线配置示例

如果你要跑当前增强版 tracking 主线，请显式切换到 `tracking_final.yaml`，例如：

```bash
python tools/infer.py \
  --config configs/main/tracking_final.yaml \
  --weights outputs/experiments/full_project/weights/best.pt \
  --source_rgb path/to/rgb_sequence \
  --source_ir path/to/ir_sequence \
  --save_dir outputs/tracking_final
```

说明：

- `tracking_base.yaml` 是最小 tracking-by-detection 对照入口。
- `tracking_final.yaml` 是当前推荐的增强版 tracking 主入口。
- `tracking_eval.yaml` 是配套的离线评估入口，用于评估已有 `tracking_results.json`，不是新的 tracking 主线。
- `configs/archive/tracking/` 下的中间阶段配置只保留给历史实验或兼容，不再作为 README 主入口。

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
- 每个 epoch 的训练损失，以及按 `train.eval_interval` 触发的验证指标（最后一轮强制验证）
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
- 主线默认是 `dataset.imgsz: 1024`；快速试跑时可临时降到 `960` / `800` / `640`
- 必要时暂时把 `dataloader.num_workers` 调低，先确认环境和数据链路没问题

## 当前状态与注意事项

- 当前检测主线推荐入口是 `configs/main/full_project.yaml`，对应 RGB+IR 双流、`ReliabilityAwareFusion`、`two_frame` temporal 和 OBB detection。
- `full_project` 当前默认训练参数是 `batch_size=4`、`imgsz=1024`、`epochs=300`、`patience=50`、`eval_interval=5`、`lr=0.0003`、`use_amp=False`；dataset 主线路径保持 `dataset/DroneVehicle_process`。
- detection 验证默认已经切到 GPU evaluator：`eval.evaluator=gpu` + `eval.obb_iou_backend=gpu_prob`。`gpu_prob` 是 surrogate / ProbIoU 路线，不是 exact polygon IoU；CPU 现在是 reference / fallback / analysis 路径，不再是并列主线。
- `configs/main/` 下的其他检测配置主要用于模块消融和专项实验。
- 当前 tracking 主入口只突出 `configs/main/tracking_base.yaml` 和 `configs/main/tracking_final.yaml`；`configs/main/tracking_eval.yaml` 是配套的离线评估配置。
- detection 侧 temporal memory 仅保留为历史实验 / 兼容路径，不再代表检测主线；tracking 侧 memory 仍然是 `tracking_final` 的有效能力。
- `configs/archive/tracking/` 和旧 `configs/exp_tracking_*.yaml` 主要用于兼容和历史实验复现，不建议新用户优先使用。
- `configs/archive/` 和旧 `configs/exp_*.yaml` 主要用于兼容和历史实验复现，不建议新用户优先使用。
- `tools/infer.py` 是当前推荐的推理入口；`tools/track.py` 更接近兼容或较早期的跟踪封装。
- `docs/` 下的模板文档适合整理实验、答辩或交付材料，但不代替训练入口说明。

## 补充文档

- 配置说明：[`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md)
- GPU evaluator 迁移阶段记录：[`docs/GPU_EVAL_STAGE3_GPU_EVALUATOR.md`](docs/GPU_EVAL_STAGE3_GPU_EVALUATOR.md)、[`docs/GPU_EVAL_STAGE4_PARITY.md`](docs/GPU_EVAL_STAGE4_PARITY.md)、[`docs/GPU_EVAL_STAGE5_DEFAULT_SWITCH.md`](docs/GPU_EVAL_STAGE5_DEFAULT_SWITCH.md)、[`docs/GPU_EVAL_STAGE6_CLEANUP.md`](docs/GPU_EVAL_STAGE6_CLEANUP.md)
- 实验计划模板：[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
- 技术方案模板：[`docs/TECHNICAL_PLAN_TEMPLATE.md`](docs/TECHNICAL_PLAN_TEMPLATE.md)
- PPT 大纲：[`docs/PPT_OUTLINE.md`](docs/PPT_OUTLINE.md)
- 消融表模板：[`docs/ABLATION_TABLE_TEMPLATE.md`](docs/ABLATION_TABLE_TEMPLATE.md)
- Tracking 结果模板：[`docs/RESULTS_TRACKING_TEMPLATE.md`](docs/RESULTS_TRACKING_TEMPLATE.md)
