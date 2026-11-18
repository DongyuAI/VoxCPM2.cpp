# VoxCPM2 在 GTX 1070 Pascal 上的适配与特化优化实测报告

测试日期：2026-11-18
硬件：NVIDIA GTX 1070 8GB（sm_61 / Pascal），CUDA 12.6
引擎：audio.cpp 0.9.5（git `61c7838b`，2026-11-14，MSVC 19.44 Release）
模型：audio.cpp 官方 `VoxCPM2-GGUF/voxcpm2-q8_0.gguf`，2,955,007,488 字节（ModelScope 镜像下载）
硬约束：全部固定 `num_inference_steps=6`、`seed=1234`、同一段 101 字中文，**不降精度、不降步数**

---

## 一、结论速览

| 问题 | 答案 |
|---|---|
| VoxCPM2 有没有适配？ | **有**。audio.cpp 0.9.5 原生支持 `family=voxcpm2`，覆盖 tts/clone/design、离线/流式、48kHz 输出；预编译版已确认 CUDA 后端可用。VoxCPM.cpp 不支持 VoxCPM2（上游停在 2026-07-14）。 |
| 有没有特化优化？ | **有**。① `perf_mode=cuda_perf`（QKV / gate-up 权重打包融合 matmul）；② 官方 `GGML_CUDA_GRAPHS_PRE_AMPERE` 环境变量解锁 Pascal CUDA Graph；③ `prompt_cache_slots` 参考音频 latent 缓存；④ `audiovae_tile_frames` 分块解码解决 8GB 显存 OOM。 |
| 优化是否牺牲质量？ | **否**。三组 A/B 输出 WAV 的 SHA256 完全相同：`c360888d48cefd9b9619fb6853e54d285084a32f8f85051b72a4e2fa0a86db2a`。tile 分块与 host arena 缩减经单独验证也逐字节一致（`bc9db498...`）。 |
| 100 字实测 | 音频 22.88s / 48kHz 单声道；wall 9.0–9.4s；RTF 0.392–0.410；峰值显存 **5613 MiB**（基线 1260 MiB）。 |

---

## 二、关键坑：模型格式不兼容（已解决）

最初从 `DennisHuang648/VoxCPM2-GGUF` 下载的是 llama.cpp-omni 风格的**分离双文件**：

- `VoxCPM2-BaseLM-Q8_0.gguf`（1.73GB）
- `VoxCPM2-Acoustic-F16.gguf`（1.83GB）

audio.cpp **无法加载**这种格式。其 `model_specs/voxcpm2.json` 明确要求单个合并 GGUF（内部同时含 `weights.*` 与 `audiovae_weights.*`），目录里有多个 GGUF 且无 `model.gguf` 时会直接报 ambiguous 拒绝。

正确模型来自 `audio-cpp/audio.cpp-gguf` 官方仓库（ModelScope 有完整镜像）：

```
VoxCPM2-GGUF/voxcpm2-q8_0.gguf   2,955,007,488 bytes
```

新格式 standalone GGUF 内嵌 model spec，不需要额外 config/tokenizer sidecar。

---

## 三、第一次失败：默认配置 OOM（已解决）

默认参数下冒烟测试直接失败：

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 160.20 MiB on device 0:
cudaMalloc failed: out of memory
audiocpp_cli failed: failed to allocate VoxCPM2 AudioVAE graph
```

日志显示权重本身只占 2.81GB：

```
voxcpm2.model.weights.buffer_mb        2455.52
voxcpm2.audiovae.weights.buffer_mb      357.68
```

说明是 AudioVAE decoder 一次性申请大 workspace + 桌面程序常驻占用（约 1.27GB）叠加导致 8GB 撑爆。

**解法**（数学等价，已验证输出哈希一致）：

```
--session-option audiovae_tile_frames=1          # 因果分块解码，避免一次性大 workspace
--session-option dit_graph_arena_mb=256           # 以下四项是 host graph-metadata arena，
--session-option lm_step_graph_arena_mb=256       # 不是 device workspace，缩小不影响数值
--session-option audiovae_graph_arena_mb=256
--session-option audiovae_encoder_graph_arena_mb=256
```

对比：

| 配置 | decode 耗时 | 输出 |
|---|---|---|
| `tile_frames=8`（默认 arena） | 922.99 ms | 138,284 B |
| `tile_frames=1` + 缩小 arena | **71.86 ms** | 138,284 B（SHA256 相同） |

---

## 四、三组 A/B 结果（100 字 / 6 步 / seed 1234）

| 组 | 配置 | metrics.wall | RTF | generator | VAE decode | 输出 SHA256 |
|---|---|---|---|---|---|---|
| A_base | `perf_mode=off`，graph 关 | 9388.84 ms | 0.410352 | 8560.13 ms | 806.93 ms | `c360888d…` |
| B_packed | `perf_mode=cuda_perf`，graph 关 | 9114.17 ms | 0.398347 | — | — | `c360888d…` |
| C_packedgr | `cuda_perf` + Pascal CUDA Graph | **8975.29 ms** | **0.392277** | 8162.95 ms | 798.02 ms | `c360888d…` |

- `cuda_perf` 单独收益：wall −274ms（−2.9%），RTF −0.012
- CUDA Graph 叠加收益：wall −139ms（−1.5%），RTF −0.006
- 合计提速：wall −414ms（−4.4%），RTF 0.4104 → 0.3923

CUDA Graph 在 audio.cpp 上**确实启用成功**（日志 `CUDA graph warmup complete`），与之前 VoxCPM.cpp 上 graph 反而变慢的结果不同——原因是 audio.cpp 内嵌的 ggml 更新（150 个 CUDA 源文件 vs VoxCPM.cpp 的 133 个，含 small_k 等新 kernel），并提供了官方开关 `GGML_CUDA_GRAPHS_PRE_AMPERE`，无需手改源码 gate。

瓶颈定位：generator（LM + DiT flow sampling）占 8.5s ≈ 91%，VAE decode 仅 0.8s。所以算子融合/graph 这类优化的相对收益受限于生成阶段，符合预期。

---

## 五、步数参数真实性验证

用 `--request-option num_inference_steps=N` 短文本对照（仅诊断，不改基准）：

| steps | metrics.wall | RTF |
|---|---|---|
| 1 | 503.28 ms | 0.2860 |
| 6 | 757.86 ms | 0.5263 |

耗时随步数线性增长，确认 `num_inference_steps=6` 真实生效。另有约 3.5s 固定加载开销（模型上传 CUDA + graph 构建），这是短文本 RTF 偏高、长文本 RTF 偏低的原因。

---

## 六、显存实测

后台 50ms 间隔采样 300 个点（合成期间）：

| 指标 | 数值 |
|---|---|
| 基线（桌面常驻） | 1260 MiB |
| 峰值 | **5613 MiB** |
| 模型净增 | 约 4353 MiB |
| 系统内存峰值 RSS | 3639 MB |
| Windows commit footprint | 14814 MB |

权重明细：`weights.buffer_mb 2455.52` + `audiovae.weights.buffer_mb 357.68` + decoder workspace `1340.63`（tile 后实际按需分配）。

8GB 卡可稳定运行，余量约 2.5GB。

---

## 七、与 VoxCPM.cpp（1.0）的对比

| 维度 | VoxCPM.cpp q8_0 6步 | VoxCPM2 audio.cpp q8_0 6步 |
|---|---|---|
| 参数规模 | 0.5B 级 | 2B 级 |
| 采样率 | 16kHz | 48kHz |
| 100字合成 wall | 5.23–5.64 s | 8.98–9.39 s |
| 音频时长 | 10.24 s | 22.88 s |
| RTF | 0.527 | 0.392–0.410 |
| 峰值显存 | ~3.3 GB | 5.61 GB |
| CUDA Graph | 手改 patch 后反而慢 5%，弃用 | 官方开关启用成功，快 1.5% |
| 算子融合 | 内建 fused_front_half | perf_mode=cuda_perf（QKV/gate-up packing） |
| 参考音频缓存 | 需 server VoiceStore 注册 | prompt_cache_slots（默认 1） |

VoxCPM2 更慢但音频时长也更长（48kHz 高质量）；RTF 反而更低，单位音频生成效率更好。

---

## 八、版本管理

VoxCPM2 工作区 `F:/Tsinglin_Voice/TTS/VoxCPM2` 为独立 git 仓库，关键 commit：

```
a4abd65 feat(bench): 正式三组A/B改用已验证数值等价的低显存配置(tile1+缩小host arena)
403c8ae feat(bench): 加入Pascal CUDA Graph三组A/B矩阵(base/packed/packed+graph)，锁6步固定seed
510544b chore: gitignore 排除 GGUF 模型与 incomplete 临时文件
4319e70 feat(voxcpm2): 改用ModelScope下载audio.cpp官方单文件GGUF，修正A/B脚本指向Q8_0并锁6步
590f7de fix(download): 合并后保留分片，新增断点续合并脚本避免批量删除拦截
6abf917 chore: VoxCPM2 实测工作区初始化
```

模型二进制与日志由 `.gitignore` 排除，不入库。

---

## 九、可复现命令

```powershell
# 冒烟（低显存配置）
& "F:\Tsinglin_Voice\TTS\VoxCPM2\_smoke.ps1"

# 三组 A/B
& "F:\Tsinglin_Voice\TTS\VoxCPM2\_ab_perf.ps1"

# 显存精确采样
& "F:\Tsinglin_Voice\TTS\VoxCPM2\_vram_measure.ps1"
```

核心 CLI：

```bash
audiocpp_cli --task tts --family voxcpm2 \
  --model models_ms/VoxCPM2-GGUF/voxcpm2-q8_0.gguf --backend cuda \
  --text "..." --out out.wav --threads 8 --seed 1234 \
  --num-inference-steps 6 \
  --session-option perf_mode=cuda_perf \
  --session-option audiovae_tile_frames=1 \
  --session-option dit_graph_arena_mb=256 \
  --session-option lm_step_graph_arena_mb=256 \
  --session-option audiovae_graph_arena_mb=256 \
  --session-option audiovae_encoder_graph_arena_mb=256 \
  --metrics --log-file run.log
# Pascal CUDA Graph: 额外设置环境变量 GGML_CUDA_GRAPHS_PRE_AMPERE=1
```
