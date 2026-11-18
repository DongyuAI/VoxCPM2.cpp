#!/usr/bin/env bash
# A/B: Pascal MMVQ rows_per_cuda_block 2 -> 1 (Q8_0, ncols_dst=8)
# 硬约束: 6步 / seed1234 / Q8_0 / cuda_perf / graph ON / 100字
# 对比: wav SHA256 必须完全一致
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
  local name="$1" exe="$2"
  local out="$ROOT/TTS/VoxCPM2/rpb1_${name}.wav"
  local log="$ROOT/TTS/VoxCPM2/rpb1_${name}.log"
  rm -f "$out" "$log"
  echo "=== RUN $name ($exe) ==="
  "$exe" "${COMMON[@]}" --out "$out" --log-file "$log" > "$ROOT/TTS/VoxCPM2/rpb1_${name}.stdout" 2>&1
  local rc=$?
  local sha="NONE"
  [ -f "$out" ] && sha=$(sha256sum "$out" | cut -d' ' -f1)
  local compute="0"
  [ -f "$log" ] && compute=$(grep -o 'voxcpm2\.dit\.compute_ms [0-9.]*' "$log" | awk '{s+=$2} END{printf "%.1f",s}')
  local rtf="NA"
  [ -f "$log" ] && rtf=$(grep -o 'metrics\.rtf=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
  local gen="NA"
  [ -f "$log" ] && gen=$(grep -o 'metrics\.wall_ms=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
  echo "RESULT name=$name rc=$rc SHA=$sha DiT_compute_ms=$compute RTF=$rtf wall=$gen"
}

run base "$BIN/audiocpp_mmqy32.exe"
run opt  "$BIN/audiocpp_cli.exe"

sha_a=$(sha256sum "$ROOT/TTS/VoxCPM2/rpb1_base.wav" 2>/dev/null | cut -d' ' -f1)
sha_b=$(sha256sum "$ROOT/TTS/VoxCPM2/rpb1_opt.wav" 2>/dev/null | cut -d' ' -f1)
echo "SHA_EQUAL=$([ "$sha_a" = "$sha_b" ] && echo YES || echo NO)"
echo "BASE_SHA=$sha_a"
echo "OPT_SHA=$sha_b"
