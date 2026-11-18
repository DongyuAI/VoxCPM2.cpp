"""续合并：把现存 partN 追加到已部分合并的 gguf，不删除任何分片（避免触发批量删除拦截）。"""
import os, re, sys

d = r"F:\Tsinglin_Voice\TTS\VoxCPM2\models"
CHUNK = 16 * 1024 * 1024
FILES = [
    ("VoxCPM2-BaseLM-Q8_0.gguf", 1727309920),
    ("VoxCPM2-Acoustic-F16.gguf", 1825096352),
]

for name, size in FILES:
    path = os.path.join(d, name)
    cur = os.path.getsize(path) if os.path.exists(path) else 0
    if cur == size:
        print(f"[SKIP] {name} complete", flush=True)
        continue
    # 找出所有现存分片编号
    parts = {}
    for f in os.listdir(d):
        m = re.fullmatch(re.escape(name) + r"\.part(\d+)", f)
        if m:
            parts[int(m.group(1))] = os.path.join(d, f)
    if not parts:
        print(f"[FAIL] {name}: no parts left, current={cur}", flush=True)
        continue
    # 按编号顺序，从 cur 对应的 offset 继续写
    # 每个 part i 覆盖 [i*CHUNK, (i+1)*CHUNK)
    start_part = cur // CHUNK
    mode = "ab" if cur > 0 else "wb"
    if cur % CHUNK != 0:
        print(f"[WARN] {name}: cur={cur} not chunk-aligned (rem={cur%CHUNK})", flush=True)
    print(f"[MERGE] {name}: cur={cur}, start_part={start_part}, have_parts={min(parts)}-{max(parts)} count={len(parts)}", flush=True)
    with open(path, mode) as out:
        for i in sorted(parts):
            expected_off = i * CHUNK
            if expected_off < cur - CHUNK:  # 已合并过
                continue
            with open(parts[i], "rb") as f:
                data = f.read()
            # 若该分片已被部分写入目标，按 cur 内偏移跳过
            skip = max(0, cur - expected_off)
            if skip >= len(data):
                continue
            out.write(data[skip:])
            cur = expected_off + len(data)
    final = os.path.getsize(path)
    ok = final == size
    print(f"[{'OK' if ok else 'PARTIAL'}] {name}: {final}/{size}", flush=True)
print("DONE", flush=True)
