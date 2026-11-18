#!/usr/bin/env bash
# 诊断：一次 100字/6步 运行，同时采 GPU 利用率 + matmul 形状分布 + DiT graph 节点数
set -u
ROOT="F:/Tsinglin_Voice"
CACHE="$ROOT/.cache"
export TMP="$CACHE/audiocpp-temp" TEMP="$CACHE/audiocpp-temp"
export MODELSCOPE_CACHE="$CACHE/modelscope" HF_HOME="$CACHE/huggingface" XDG_CACHE_HOME="$CACHE/xdg"
export VOXCPM2_DIT_PROFILE=1
export GGML_CUDA_LOG_MATMUL=1
export GGML_CUDA_GRAPHS_PRE_AMPERE=1
BIN="$ROOT/TTS/audio.cpp/build/windows-cuda-release/bin"
export PATH="$BIN:$PATH"
MODEL="$ROOT/TTS/VoxCPM2/models_ms/VoxCPM2-GGUF/voxcpm2-q8_0.gguf"
TEXT="大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
OUT="$ROOT/TTS/VoxCPM2/probe.wav"
LOG="$ROOT/TTS/VoxCPM2/probe.log"
SMI="$ROOT/TTS/VoxCPM2/probe_smi.csv"
rm -f "$OUT" "$LOG" "$SMI"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw --format=csv,noheader -lms 250 -f "$SMI" &
SMI_PID=$!

"$BIN/audiocpp_cli.exe" \
  --task tts --family voxcpm2 --model "$MODEL" --backend cuda \
  --text "$TEXT" --out "$OUT" --threads 8 --seed 1234 \
  --num-inference-steps 6 --metrics --log-file "$LOG" \
  --session-option perf_mode=cuda_perf \
  --session-option audiovae_tile_frames=1 \
  --session-option dit_graph_arena_mb=256 \
  --session-option lm_step_graph_arena_mb=256 \
  --session-option audiovae_graph_arena_mb=256 \
  --session-option audiovae_encoder_graph_arena_mb=256 \
  > "$ROOT/TTS/VoxCPM2/probe.stdout" 2> "$ROOT/TTS/VoxCPM2/probe.stderr"
RC=$?
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo "RC=$RC"
echo "=== matmul 形状分布 ==="
grep 'MATMUL-SHAPE-PROBE' "$ROOT/TTS/VoxCPM2/probe.stderr" | sort -u
echo "=== DiT graph nodes ==="
grep 'dit.graph_nodes' "$LOG" | head -3
echo "=== GPU 利用率汇总 ==="
awk -F', *' 'NR>0 {g+=$2+0; m+=$3+0; n++; if($2+0>mx)mx=$2+0} END {if(n>0) printf "samples=%d avg_gpu_util=%.1f%% avg_mem_util=%.1f%% peak_gpu_util=%.1f%%\n",n,g/n,m/n,mx}' "$SMI"
echo "=== metrics ==="
grep '^metrics\.' "$ROOT/TTS/VoxCPM2/probe.stdout"
