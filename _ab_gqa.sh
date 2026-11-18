#!/usr/bin/env bash
# 严格 A/B: 同一个 exe, 唯一差异 = VOXCPM2_GQA_CONCAT env 开关
#   concat = 原 15次 Slice+Concat (363 CONCAT 节点)
#   repeat = 新 reshape+repeat (SHA-safe, 预期 ~1 repeat 节点/层)
# 硬约束: 6步 / seed1234 / Q8_0 / cuda_perf / graph ON / 100字
set -u
ROOT="F:/Tsinglin_Voice"
CACHE="$ROOT/.cache"
export TMP="$CACHE/audiocpp-temp" TEMP="$CACHE/audiocpp-temp"
export MODELSCOPE_CACHE="$CACHE/modelscope" HF_HOME="$CACHE/huggingface" XDG_CACHE_HOME="$CACHE/xdg"
export VOXCPM2_DIT_PROFILE=1
export GGML_CUDA_GRAPHS_PRE_AMPERE=1
BIN="$ROOT/TTS/audio.cpp/build/windows-cuda-release/bin"
export PATH="$BIN:$PATH"
MODEL="$ROOT/TTS/VoxCPM2/models_ms/VoxCPM2-GGUF/voxcpm2-q8_0.gguf"
TEXT="大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

COMMON=(
  --task tts --family voxcpm2 --model "$MODEL" --backend cuda
  --text "$TEXT" --threads 8 --seed 1234
  --num-inference-steps 6 --metrics
  --session-option perf_mode=cuda_perf
  --session-option audiovae_tile_frames=1
  --session-option dit_graph_arena_mb=256
  --session-option lm_step_graph_arena_mb=256
  --session-option audiovae_graph_arena_mb=256
  --session-option audiovae_encoder_graph_arena_mb=256
)

run() {
  local name="$1"
  local out="$ROOT/TTS/VoxCPM2/gqa_${name}.wav"
  local log="$ROOT/TTS/VoxCPM2/gqa_${name}.log"
  local err="$ROOT/TTS/VoxCPM2/gqa_${name}.stderr"
  rm -f "$out" "$log" "$err"
  echo "=== RUN $name ==="
  "$BIN/audiocpp_cli.exe" "${COMMON[@]}" --out "$out" --log-file "$log" \
    > "$ROOT/TTS/VoxCPM2/gqa_${name}.stdout" 2> "$err"
  local rc=$?
  local sha="NONE"
  [ -f "$out" ] && sha=$(sha256sum "$out" | cut -d' ' -f1)
  local compute="0"
  [ -f "$log" ] && compute=$(grep -o 'voxcpm2\.dit\.compute_ms [0-9.]*' "$log" | awk '{s+=$2} END{printf "%.1f",s}')
  local nodes="NA"
  [ -f "$log" ] && nodes=$(grep -o 'voxcpm2\.dit\.graph_nodes [0-9-]*' "$log" | head -1 | awk '{print $2}')
  local concat="NA"
  [ -f "$err" ] && concat=$(grep 'DIT-NODE-HIST.*CONCAT' "$err" | sed 's/.*count=//' | head -1)
  local rtf="NA"
  [ -f "$log" ] && rtf=$(grep -o 'voxcpm2\.dit\.compute_ms' "$log" | wc -l)
  echo "RESULT name=$name rc=$rc SHA=$sha DiT_compute_ms=$compute nodes=$nodes CONCAT=$concat"
}

export VOXCPM2_GQA_CONCAT=1
run concat
unset VOXCPM2_GQA_CONCAT
run repeat

sha_a=$(sha256sum "$ROOT/TTS/VoxCPM2/gqa_concat.wav" 2>/dev/null | cut -d' ' -f1)
sha_b=$(sha256sum "$ROOT/TTS/VoxCPM2/gqa_repeat.wav" 2>/dev/null | cut -d' ' -f1)
echo "SHA_EQUAL=$([ "$sha_a" = "$sha_b" ] && echo YES || echo NO)"
echo "CONCAT_SHA=$sha_a"
echo "REPEAT_SHA=$sha_b"
