# 项目说明 (含安装部署、实验复现指令)
# 🚁 UAV MultiModal OBB Detection 

本项目是一个面向无人机（UAV）全天候航拍场景的**工业级双模态（RGB+IR）旋转框（OBB）目标检测框架**。
采用极致解耦的架构设计，支持配置驱动（Config-Driven），一键完成训练、评测与 TensorRT 部署导出。

## ✨ 核心特性 (Features)
- **非对称双流架构**: 高效融合可见光与红外特征。
- **可插拔融合算子**: 支持 `DualStreamFusion`, `SimpleConcat`, `JCA` 等一键切换（消融实验利器）。
- **动态极小目标分配器**: 针对无人机极小目标设计的 Dynamic Tiny-OBB Assigner。
- **高阶数据增强管道**: 内置防冲突复制粘贴 (CMCP)、难度感知恶劣天气模拟 (Curriculum Learning) 与模态 Dropout。
- **工业级生命周期引擎**: 纯粹的 `Trainer/Evaluator` 结合 Callbacks 机制，支持 DDP 多卡分布式训练与 AMP 混合精度。

## ⚙️ 快速开始 (Quick Start)

### 1. 环境安装
```bash
# 创建虚拟环境
conda create -n uav_det python=3.9
conda activate uav_det

# 安装依赖项
pip install -r requirements.txt

# 以开发者模式安装当前项目 (极其重要，解决 src 导入问题)
pip install -e .
2. 模型训练 (Train)
支持完全通过 YAML 配置文件掌控全局超参数。

Bash
# 单卡训练
python tools/train.py --config configs/default.yaml

# 多卡分布式训练 (例如 4 卡)
bash scripts/dist_train.sh 4 configs/default.yaml
3. 消融实验 (Ablation Study)
无需修改 Python 代码，只需切换配置文件即可一键运行消融实验：

Bash
bash scripts/run_ablation.sh
4. 推理与验证 (Infer & Val)
Bash
# 验证模型精度
python tools/val.py --config configs/default.yaml --weights outputs/weights/best.pt

# 单张图片可视化推理
python tools/infer.py --config configs/default.yaml --weights outputs/weights/best.pt \
    --source_rgb test_rgb.jpg --source_ir test_ir.jpg --heatmap
5. 工业级模型导出 (Export to ONNX)
将模型剥离训练图，导出为极致精简的纯净 ONNX，为 TensorRT/NCNN 部署铺路。

Bash
python tools/export.py --config configs/default.yaml --weights outputs/weights/best.pt --half

---

**大功告成！**

现在，请打开你的终端，在项目根目录下运行这条极其重要的命令：
```bash
pip install -e .