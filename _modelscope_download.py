"""通过 ModelScope 下载 audio.cpp 官方 VoxCPM2 单文件 Q8_0 GGUF。

audio.cpp 只接受单个合并 GGUF（内部含 weights.* + audiovae_weights.*），
DennisHuang648 的 BaseLM+Acoustic 双文件是 llama.cpp-omni 格式，audio.cpp 无法加载。

用法: C:/Python314/python.exe _modelscope_download.py
"""
import os
import sys
import time
from modelscope.hub.snapshot_download import snapshot_download

MODEL_ID = "audio-cpp/audio.cpp-gguf"
PATTERN = "VoxCPM2-GGUF/voxcpm2-q8_0.gguf"
OUT_DIR = r"F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] modelscope snapshot_download {MODEL_ID} :: {PATTERN}", flush=True)
t0 = time.time()
path = snapshot_download(
    model_id=MODEL_ID,
    local_dir=OUT_DIR,
    allow_patterns=[PATTERN],
    max_workers=8,
)
print(f"[DONE] root={path} elapsed={time.time()-t0:.1f}s", flush=True)
for root, dirs, files in os.walk(path):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p} {os.path.getsize(p)}", flush=True)
