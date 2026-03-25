# 批量跑实验 (修改配置参数)
#!/bin/bash
# ============================================================================
# 多模态极小目标 OBB - 多卡分布式训练启动脚本 (DDP)
# 用法: bash scripts/dist_train.sh <GPU数量> [配置文件路径] [端口号]
# 示例: bash scripts/dist_train.sh 4 configs/model/yolov8_dual_crossattn.yaml
# ============================================================================

NUM_GPUS=$1
CONFIG=${2:-"configs/default.yaml"}
PORT=${3:-29500} # 默认端口，如果你在同一台机器跑多个任务，请修改此端口防止冲突

if [ -z "$NUM_GPUS" ]; then
    echo "❌ 错误: 必须指定 GPU 数量！"
    echo "💡 用法: bash scripts/dist_train.sh <GPU数量> [配置文件路径] [端口号]"
    exit 1
fi

echo "🚀 正在点燃分布式引擎 (DDP)..."
echo "🖥️  调用 GPU 数量: ${NUM_GPUS}"
echo "📄 挂载配置文件: ${CONFIG}"
echo "----------------------------------------------------------"

# 使用 PyTorch 官方推荐的 torchrun 启动 DDP
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    tools/train.py --config $CONFIG

echo "----------------------------------------------------------"
echo "✅ 分布式训练任务结束！"