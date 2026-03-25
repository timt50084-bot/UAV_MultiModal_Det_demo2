# UAV 多模态旋转框检测项目

一个面向 UAV 场景的 RGB + IR 双模态 OBB（Oriented Bounding Box，旋转框）检测工程。项目采用配置驱动方式组织训练、验证、推理、导出与实验扩展，当前代码已包含可插拔融合模块、面向小目标的分配器、双帧/短时序模块、真实多模态退化增强以及任务导向评测指标。

## 项目简介

本项目聚焦无人机场景下的小目标、多模态和旋转框检测问题，目标是提供一套可训练、可验证、可推理、可导出的工程化基线，并支持逐步扩展新的 fusion、temporal、assigner、augmentation 与 metrics 模块。

当前代码状态对应的核心能力包括：
- RGB + IR 双流 backbone
- 可插拔融合模块
- OBB decoupled detection head
- 面向 tiny object 的 assigner
- two-frame temporal refinement 与 short-term temporal memory
- realistic multimodal augmentation
- task-specific evaluation metrics

其中部分近期新增能力属于可选或实验性扩展，代码已经存在于当前仓库中，可通过独立配置文件启用。

## 核心特性

- 双流多模态骨干网络：`src/model/backbones/dual_backbone.py`
- 可插拔融合模块：`DualStreamFusion`、`SimpleConcatFusion`、`RDMFusion`、`ReliabilityAwareFusion`
- OBB 检测头：`src/model/heads/obb_decoupled_head.py`
- Tiny-OBB assigner：支持角度一致性、小目标保护、细长目标保护
- 时序模块：支持 `off / two_frame / memory` 三种模式
- 多模态增强：包含 CMCP、MRRE、天气模拟、模态 dropout、跨模态错位、传感器退化增强
- 任务导向评测：支持 `mAP_50`、`mAP_S`、`Recall_S`、`Precision_S`，可选 `CrossModalRobustness_*`、`TemporalStability`
- 配置驱动：默认配置位于 `configs/default.yaml`，实验配置位于 `configs/exp_*.yaml` 与 `configs/model/*.yaml`

## 项目结构

当前仓库的主要目录如下：

```text
UAV_MultiModal_Det_demo2/
├─ configs/
│  ├─ default.yaml
│  ├─ exp_angle_aware_assigner.yaml
│  ├─ exp_realistic_multimodal_aug.yaml
│  ├─ exp_reliability_fusion.yaml
│  ├─ exp_task_metrics.yaml
│  ├─ exp_temporal_memory.yaml
│  ├─ dataset/
│  └─ model/
├─ scripts/
│  ├─ dist_train.sh
│  └─ run_ablation.sh
├─ src/
│  ├─ data/
│  ├─ engine/
│  ├─ loss/
│  ├─ metrics/
│  ├─ model/
│  ├─ registry/
│  ├─ tracking/
│  └─ utils/
├─ tests/
├─ tools/
│  ├─ train.py
│  ├─ val.py
│  ├─ infer.py
│  ├─ export.py
│  └─ track.py
├─ README.md
├─ requirements.txt
└─ setup.py
```

## 环境安装

建议使用独立虚拟环境。

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd UAV_MultiModal_Det_demo2
```

### 2. 创建虚拟环境

使用 conda：

```bash
conda create -n uav-mm-obb python=3.9
conda activate uav-mm-obb
```

或使用 `venv`：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 开发模式安装

```bash
pip install -e .
```

### 5. 最小安装自检

```bash
python -c "from src.utils.config import load_config; print('ok')"
```

如果需要模型导出相关能力，可额外安装：

```bash
pip install onnx onnxsim onnxruntime
```

## 数据准备

默认数据组织方式由 `src/data/datasets/drone_rgb_ir.py` 定义，目录结构应与下列形式一致：

```text
<dataset_root>/
├─ train/
│  ├─ images/
│  │  ├─ img/
│  │  └─ imgr/
│  └─ labels/
│     └─ merged/
└─ val/
   ├─ images/
   │  ├─ img/
   │  └─ imgr/
   └─ labels/
      └─ merged/
```

标签格式为每行一个目标：

```text
cls cx cy w h theta
```

其中坐标为归一化形式，`theta` 为旋转角。

默认数据路径在 `configs/default.yaml` 中配置为示例值：

```yaml
dataset:
  root_dir: 'data/drone_rgb_ir'
```

使用前请按本地数据路径修改 `dataset.root_dir`。

## 训练方法

### 基础训练

```bash
python tools/train.py --config configs/default.yaml
```

### 使用实验配置训练

```bash
python tools/train.py --config configs/exp_reliability_fusion.yaml
python tools/train.py --config configs/exp_angle_aware_assigner.yaml
python tools/train.py --config configs/exp_temporal_memory.yaml
python tools/train.py --config configs/exp_realistic_multimodal_aug.yaml
python tools/train.py --config configs/exp_task_metrics.yaml
```

### 分布式训练脚本

仓库中保留了 `scripts/dist_train.sh`，可作为单机多卡启动脚本参考：

```bash
bash scripts/dist_train.sh 4 configs/default.yaml
```

说明：该脚本存在于当前仓库中，但实际可用性取决于本地 CUDA 与 `torch.distributed` 环境。

## 验证与评估

### 基础验证

```bash
python tools/val.py --config configs/default.yaml --weights outputs/weights/best.pt
```

### 启用任务导向指标验证

```bash
python tools/val.py --config configs/exp_task_metrics.yaml --weights outputs/weights/best.pt
```

当前评测结果字典兼容以下字段：
- `mAP_50`
- `mAP_S`
- `Recall_S`
- `Precision_S`
- `CrossModalRobustness_RGBDrop`（启用时）
- `CrossModalRobustness_IRDrop`（启用时）
- `TemporalStability`（可计算时）

说明：
- `Recall_S`、`Precision_S`、`TemporalStability` 越大越好。
- `CrossModalRobustness_RGBDrop`、`CrossModalRobustness_IRDrop` 表示相对基线的性能下降幅度，越小越好。
- 若没有可用序列信息，`TemporalStability` 会自动跳过。

## 推理、跟踪与导出

### 单张图像推理

```bash
python tools/infer.py \
  --config configs/default.yaml \
  --weights outputs/weights/best.pt \
  --source_rgb path/to/rgb.jpg \
  --source_ir path/to/ir.jpg
```

如需可视化 attention / heatmap：

```bash
python tools/infer.py \
  --config configs/default.yaml \
  --weights outputs/weights/best.pt \
  --source_rgb path/to/rgb.jpg \
  --source_ir path/to/ir.jpg \
  --heatmap
```

### 序列跟踪

```bash
python tools/track.py \
  --config configs/default.yaml \
  --weights outputs/weights/best.pt \
  --source_rgb_dir path/to/rgb_frames \
  --source_ir_dir path/to/ir_frames
```

### 导出 ONNX

```bash
python tools/export.py --config configs/default.yaml --weights outputs/weights/best.pt
```

可选参数：

- `--half`：可用 CUDA 时导出 FP16
- `--dynamic`：启用动态 batch

## 配置说明

### 默认配置

- `configs/default.yaml`：全局默认训练、验证与推理配置

### 模块切换入口

- Fusion：`cfg.model.fusion_att_type` 或 `cfg.model.fusion.type`
- Temporal：`cfg.model.temporal_enabled` 或 `cfg.model.temporal.mode`
- Assigner：`cfg.assigner`
- Augmentation：`cfg.dataset.aug_cfg`
- Extra metrics：`cfg.eval.extra_metrics`

### 已提供的实验配置

- `configs/exp_reliability_fusion.yaml`
- `configs/exp_angle_aware_assigner.yaml`
- `configs/exp_temporal_memory.yaml`
- `configs/exp_realistic_multimodal_aug.yaml`
- `configs/exp_task_metrics.yaml`

## 已实现模块概览

### 已实现的基础模块

- Asymmetric dual backbone
- Enhanced neck
- OBB decoupled head
- OBB NMS 与基础评测

### 当前仓库已落地的新增或扩展模块

- `ReliabilityAwareFusion`
  - 位置：`src/model/fusion/reliability_fusion.py`
  - 状态：可选 / 实验性融合模块

- 改进版 `DynamicTinyOBBAssigner`
  - 位置：`src/loss/assigners/target_assigner.py`
  - 状态：已兼容原有接口，新增角度一致性、小目标保护、细长目标保护

- `TemporalMemoryFusion`
  - 位置：`src/model/temporal/temporal_memory.py`
  - 状态：可选 / 实验性时序扩展模块

- realistic multimodal augmentation
  - 位置：`src/data/transforms/augmentations.py`
  - 状态：可选增强模块，包含 `CrossModalMisalignment` 与 `SensorDegradationAug`

- task-specific metrics
  - 位置：`src/metrics/task_specific_metrics.py`
  - 状态：可选评测扩展，增加小目标指标、模态鲁棒性指标和轻量时序稳定性指标

说明：这些模块都已存在于当前代码目录中，但建议通过独立实验配置逐项启用，而不是一次性全部叠加。

## 测试

当前仓库包含若干最小单元测试：

```bash
python -m unittest tests.test_reliability_fusion
python -m unittest tests.test_tiny_obb_assigner
python -m unittest tests.test_temporal_memory
python -m unittest tests.test_realistic_multimodal_aug
python -m unittest tests.test_task_specific_metrics
```

部分测试依赖 `torch` 或 `opencv-python`。若环境缺少相关依赖，测试会因导入失败而无法运行。

## 后续可扩展方向

- 更完整的多帧训练数据流
- 更强的时序一致性与视频级建模
- 更细致的多模态退化建模与鲁棒性评测
- 导出与部署链路的进一步工程化

## 说明

- 本 README 以当前仓库代码状态为准整理。
- 若后续继续引入新模块，请优先同步更新 `configs/`、`requirements.txt` 和本文中的命令示例。
