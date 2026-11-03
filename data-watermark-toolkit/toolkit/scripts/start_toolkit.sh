#!/bin/bash
# 启动 DataMark Toolkit 参数化实验服务（本机 GPU）
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/data2/renxiao/toolkit/.gpubase${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/work/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
PY="${TOOLKIT_PYTHON:-/home/renxiao/anaconda3/envs/wbench/bin/python}"
PORT="${PORT:-8765}"
exec "$PY" scripts/toolkit_server.py --port "$PORT"
