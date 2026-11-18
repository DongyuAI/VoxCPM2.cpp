#!/bin/bash
# Pascal MMQ mmq_y 优化 A/B: base(mmq_y=64) vs opt(mmq_y=8)
# 锁 6步/seed1234/100字/Q8_0/cuda_perf/graph ON, 对比 SHA + DiT compute
# 用前台直接调用（PowerShell | Out-Null 会导致进程提前返回）
set -u
BDIR="F:/Tsinglin_Voice/TTS/audio.cpp/build/windows-cuda-release/bin"
ROOT="F:/Tsinglin_Voice"
MODEL="$ROOT/TTS/VoxCPM2/models_ms/VoxCPM2-GGUF/voxcpm2-q8_0.gguf"
TEXT="大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
COMMON="--task tts --family voxcpm2 --model $MODEL --backend cuda --text $TEXT --threads 8 --seed 1234 --num-inference-steps 6 --metrics --session-option perf_mode=cuda_perf --session-option audiovae_tile_frames=1 --session-option dit_graph_arena_mb=256 --session-option lm_step_graph_arena_mb=256 --session-option audiovae_graph_arena_mb=256 --session-option audiovae_encoder_graph_arena_mb=256"
export TMP="F:/Tsinglin_Voice/.cache/audiocpp-temp"
export TEMP="F:/Tsinglin_Voice/.cache/audiocpp-temp"
export VOXCPM2_DIT_PROFILE=1
export GGML_CUDA_GRAPHS_PRE_AMPERE=1

run_one() {
  label=$1; exe=$2
  out="$ROOT/TTS/VoxCPM2/_mmqy_${label}.wav"
  log="$ROOT/TTS/VoxCPM2/_mmqy_${label}.log"
  rm -f "$out" "$log"
  echo "########## $label ##########"
  (cd "$BDIR" && timeout 240 "./${exe}" $COMMON --out "$out" --log-file "$log" > /dev/null 2>&1)
  sha=$(sha256sum "$out" 2>/dev/null | cut -c1-24)
  sum=$(C:/Python314/python.exe -c "
import re,sys
s=open(r'$log',encoding='utf-8',errors='replace').read()
c=[float(x) for x in re.findall(r'voxcpm2\.dit\.compute_ms ([0-9.]+)',s)]
calls=[int(x) for x in re.findall(r'voxcpm2\.dit\.calls ([0-9.]+)',s)]
print('%.1f %d'%(sum(c),sum(calls)))
" 2>/dev/null)
  rtf=$(grep "metrics.rtf=" "$log" 2>/dev/null | head -1)
  echo "$label: compute_sum_ms/calls = $sum | SHA = $sha | $rtf"
}

echo "=== Pascal MMQ mmq_y A/B (mmq_y=64 base vs mmq_y=8 opt) ==="
run_one base audiocpp_base.exe
run_one opt  audiocpp_cli.exe
echo "=== SHA 对比 ==="
sha_base=$(sha256sum "$ROOT/TTS/VoxCPM2/_mmqy_base.wav" 2>/dev/null | cut -d' ' -f1)
sha_opt=$(sha256sum "$ROOT/TTS/VoxCPM2/_mmqy_opt.wav" 2>/dev/null | cut -d' ' -f1)
echo "base=$sha_base"
echo "opt =$sha_opt"
[ "$sha_base" = "$sha_opt" ] && echo "SHA_EQUAL=TRUE (数值一致)" || echo "SHA_EQUAL=FALSE (数值改变!)"
