# #!/bin/bash

# USE_MEGATRON=${USE_MEGATRON:-1}
# USE_SGLANG=${USE_SGLANG:-1}

# export MAX_JOBS=32

# echo "1. install inference frameworks and pytorch they need"
# if [ $USE_SGLANG -eq 1 ]; then
#     pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
# fi
# pip install --no-cache-dir "vllm==0.11.0"

# echo "2. install basic packages"
# pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
#     "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
#     ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
#     pytest py-spy pre-commit ruff tensorboard 

# echo "pyext is lack of maintainace and cannot work with python 3.12."
# echo "if you need it for prime code rewarding, please install using patched fork:"
# echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

# pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


# echo "3. install FlashAttention and FlashInfer"
# # Install flash-attn-2.8.1 (cxx11abi=False)
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
#     pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# pip install --no-cache-dir flashinfer-python==0.3.1


# if [ $USE_MEGATRON -eq 1 ]; then
#     echo "4. install TransformerEngine and Megatron"
#     echo "Notice that TransformerEngine installation can take very long time, please be patient"
#     pip install "onnxscript==0.3.1"
#     NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
#     pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
# fi


# echo "5. May need to fix opencv"
# pip install opencv-python
# pip install opencv-fixer && \
#     python -c "from opencv_fixer import AutoFix; AutoFix()"


# if [ $USE_MEGATRON -eq 1 ]; then
#     echo "6. Install cudnn python package (avoid being overridden)"
#     pip install nvidia-cudnn-cu12==9.10.2.21
# fi

# echo "Successfully installed all packages"

# 设置 CUDA 路径 (使用 conda 环境中的 CUDA)
# 设置 CUDA 主目录
# 激活环境
# ==========================================
# Transformer Engine 编译修复脚本 (绕过 mscclpp)
# ==========================================

# 1. 确保环境激活
# conda activate verl

pip uninstall nvidia-nccl-cu12 nvidia-cudnn-cu12

echo "📦 [1/5] 正在安装标准版依赖 (确保拥有干净的头文件)..."
# 强制安装/更新官方标准包，确保我们有“好”的文件
pip install nvidia-nccl-cu12==2.27.3 nvidia-cudnn-cu12==9.10.2.21 --upgrade

echo "🔍 [2/5] 正在定位标准版头文件路径..."

# --- 定位标准 NCCL (关键步骤：专门找 nvidia/nccl 下的文件，避开坏文件) ---
STD_NCCL_FILE=$(find $CONDA_PREFIX -path "*/nvidia/nccl/include/nccl.h" | head -n 1)
if [ -z "$STD_NCCL_FILE" ]; then
    echo "❌ 错误：未找到标准版 nccl.h，请检查 pip install 是否报错。"
    return 1 2>/dev/null || exit 1
fi
STD_NCCL_DIR=$(dirname "$STD_NCCL_FILE")
echo "   ✅ 锁定标准 NCCL: $STD_NCCL_DIR"

# --- 定位 cuDNN (优先找 nvidia/cudnn 下的文件) ---
CUDNN_FILE=$(find $CONDA_PREFIX -path "*/nvidia/cudnn/include/cudnn.h" | head -n 1)
# 如果找不到，尝试全局搜
if [ -z "$CUDNN_FILE" ]; then
    CUDNN_FILE=$(find $CONDA_PREFIX -name "cudnn.h" | head -n 1)
fi
if [ -z "$CUDNN_FILE" ]; then
     echo "❌ 错误：未找到 cudnn.h"
     return 1 2>/dev/null || exit 1
fi
CUDNN_DIR=$(dirname "$CUDNN_FILE")
echo "   ✅ 锁定 cuDNN: $CUDNN_DIR"

# --- 定位 NVTX (解决最早的那个报错) ---
NVTX_FILE=$(find $CONDA_PREFIX -name "nvToolsExt.h" | head -n 1)
if [ -z "$NVTX_FILE" ]; then
    # 尝试去系统路径找
    NVTX_FILE=$(find /usr/local/cuda -name "nvToolsExt.h" 2>/dev/null | head -n 1)
fi
if [ -z "$NVTX_FILE" ]; then
     echo "⚠️  警告：未找到 nvToolsExt.h，如果报错请安装 cuda-nvtx"
     NVTX_DIR=""
else
     NVTX_DIR=$(dirname "$NVTX_FILE")
     echo "   ✅ 锁定 NVTX: $NVTX_DIR"
fi

echo "🔧 [3/5] 设置环境变量 (强制覆盖坏路径)..."

# 【核心操作】将标准 NCCL 路径放在最前面！
# 这样编译器遇到 #include <nccl.h> 时，会先用我们的，而不是你环境里那个坏的。
export CPATH=$STD_NCCL_DIR:$CUDNN_DIR:$NVTX_DIR:$CPATH

# 同时设置库路径，防止链接错误
export LD_LIBRARY_PATH=$STD_NCCL_DIR/../lib:$CUDNN_DIR/../lib:$LD_LIBRARY_PATH

echo "   当前 CPATH 开头为: $STD_NCCL_DIR"

echo "🚀 [4/5] 清理缓存并开始安装..."
# 使用 --no-cache-dir 防止 pip 使用之前失败的缓存构建

echo "✅ [5/5] 脚本执行完毕。"

# 再次尝试安装
pip3 install --no-build-isolation --no-cache-dir "transformer-engine[pytorch]"
