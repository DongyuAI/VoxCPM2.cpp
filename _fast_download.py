"""多线程分段下载 VoxCPM2 GGUF（xet CDN 支持 Range）。
用法: C:/Python314/python.exe _fast_download.py
断点续传：每个 .partN 分片独立，重跑只下未完成的。
"""
import os, sys, requests, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR = r"F:\Tsinglin_Voice\TTS\VoxCPM2\models"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("VoxCPM2-BaseLM-Q8_0.gguf", 1727309920),
    ("VoxCPM2-Acoustic-F16.gguf", 1825096352),
]
BASE = "https://hf-mirror.com/DennisHuang648/VoxCPM2-GGUF/resolve/main/"
NTHREADS = 8
CHUNK = 16 * 1024 * 1024  # 16MB 每片

session_local = threading.local()

def get_session():
    if not hasattr(session_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        session_local.s = s
    return session_local.s

def resolve_real_url(url):
    r = get_session().head(url, allow_redirects=True, timeout=30)
    return r.url

def dl_chunk(url, start, end, part_path, idx, total):
    """下载 [start,end] 到 part_path；若已存在且大小正确则跳过。"""
    want = end - start + 1
    if os.path.exists(part_path) and os.path.getsize(part_path) == want:
        return idx, want, 0  # 已完成
    s = get_session()
    last = None
    for attempt in range(6):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with s.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "wb"
                # 若服务端忽略 Range 返回 200 全量，则只取前 want 字节
                written = 0
                with open(part_path, "wb") as f:
                    for buf in r.iter_content(256 * 1024):
                        if not buf:
                            continue
                        take = min(len(buf), want - written)
                        f.write(buf[:take])
                        written += take
                        if written >= want:
                            break
                if written != want:
                    raise RuntimeError(f"short {written}/{want}")
                return idx, want, 0
        except Exception as e:
            last = e
            time.sleep(1 + attempt)
            try:
                if os.path.exists(part_path):
                    os.remove(part_path)
            except Exception:
                pass
    return idx, 0, str(last)

def download_file(name, size):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"[SKIP] {name} already complete", flush=True)
        return True
    url = resolve_real_url(BASE + name)
    print(f"[INFO] {name}: {size/1048576:.0f}MB, resolving -> {url.split('/')[2]}", flush=True)
    n_chunks = (size + CHUNK - 1) // CHUNK
    tasks = []
    for i in range(n_chunks):
        start = i * CHUNK
        end = min(start + CHUNK - 1, size - 1)
        part = os.path.join(OUT_DIR, f"{name}.part{i}")
        tasks.append((i, start, end, part))

    done_bytes = 0
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NTHREADS) as ex:
        futs = {ex.submit(dl_chunk, url, s, e, p, i, n_chunks): (i, p)
                for i, s, e, p in tasks}
        for fut in as_completed(futs):
            idx, got, err = fut.result()
            if err:
                failed.append((idx, err))
            else:
                done_bytes += got
            n_done = n_chunks - len(failed)
            elapsed = time.time() - t0
            speed = done_bytes / elapsed / 1048576 if elapsed > 0 else 0
            print(f"  {name}: {n_done}/{n_chunks} chunks, {speed:.1f} MB/s", flush=True)

    if failed:
        print(f"[FAIL] {name}: {len(failed)} chunks failed: {failed[:2]}", flush=True)
        return False

    # 合并（不删除分片——批量删除会触发安全策略拦截；分片由外部按需清理）
    print(f"[MERGE] {name} ...", flush=True)
    with open(path, "wb") as out:
        for i in range(n_chunks):
            part = os.path.join(OUT_DIR, f"{name}.part{i}")
            with open(part, "rb") as f:
                while True:
                    buf = f.read(2 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
    ok = os.path.getsize(path) == size
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {os.path.getsize(path)}/{size}", flush=True)
    return ok

if __name__ == "__main__":
    all_ok = True
    for name, size in FILES:
        all_ok = download_file(name, size) and all_ok
    print("ALL_DONE" if all_ok else "SOME_FAILED", flush=True)
