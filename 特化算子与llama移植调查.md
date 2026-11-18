# VoxCPM2 特化算子 / llama.cpp 优化移植调查

日期：2026-11-18

## 一、你问的两个问题的直接答案

### Q1：VoxCPM2.cpp 有自己的特化算子吗？

**有，但不够。** audio.cpp 里 VoxCPM2 专属实现在 `src/models/voxcpm2/`（14 个文件），特化优化包括：

| 优化 | 位置 | 覆盖率 | 状态 |
|---|---|---|---|
| `perf_mode=cuda_perf` QKV/gate-up packing | `minicpm.cpp:122,151` + `minicpm_blocks.h:169,247` | 只覆盖 **MiniCPM LM 的 q/k/v 和 gate/up 投影** | 已测，有效 −2.9% |
| DiT estimator 图 | `generator.cpp:720-968` | 标准 ggml 算子拼装，**无 VoxCPM2 专属融合算子** | 瓶颈所在（7.8s） |
| AudioVAE depthwise/transposed conv | `audiovae.cpp` | 手写权重重排（`reorder_conv_weights`），但走通用 ggml conv | 0.77s，非瓶颈 |
| `prompt_cache_slots` | `session.cpp` | 参考音频 embedding 缓存 | 有 |
| CUDA Graph | ggml 内部 `ggml_cuda_graph_get_key(cgraph)` | 靠 cgraph key 自动捕获 | 已启用，−1.5% |

**关键**：VoxCPM2 的核心耗时是 DiT + LM step 的小图反复执行，而 **DiT 没有任何专属融合算子**——它就是一堆通用 Linear/SiLU/RMSNorm/RoPE 拼出来的。这里才是算子优化的空间。

### Q2：llama.cpp 那边的优化搞过来了吗？

**还没有落地任何一项。** 我实际拉了最新 ggml（`_ggml_latest/cuda/`）逐文件对比了 audio.cpp 内嵌版本（2026-11-17 快照）。结论：

| 新 ggml 优化 | 触发条件 | VoxCPM2 + GTX1070 能用吗 |
|---|---|---|
| `halve_iters`（mmvq.cu，最新 1615 行 vs 现有 1180） | `if (table_id != MMVQ_PARAMETERS_GB10) return false;` —— **Blackwell sm_100 专用参数表** | ❌ Pascal sm_61 用不上，移植是死代码 |
| `small_k`（mmvq.cu） | 通用，但 **audio.cpp 现版本已经有** | ⚠️ 已存在，无新增 |
| sparse attention（fattn.cu，最新 838 行 vs 现有 593） | 面向超长 context 的 mask 压缩 | ❌ VoxCPM2 DiT 序列很短，用不上 |
| PDL（Programmatic Dependent Launch） | 需要 sm_90+ | ❌ Pascal 不支持 |
| MoE matvec kernel | MUL_MAT_ID / expert 权重 | ❌ VoxCPM2 不是 MoE |
| 新增 DKQ=40/72 的 flash attn case | head_dim 40/72 | ⚠️ 取决于 VoxCPM2 配置，需查 config |

所以"把 llama.cpp 优化搞过来"这句话，对 **Pascal + VoxCPM2** 这个具体组合，大部分新 kernel 是**架构不匹配**的，不能盲目搬。这不是"没空间"，而是要精准挑能落地的。

---

## 二、我犯的错：用 VoxCPM 0.5 配置套 VoxCPM2

之前那套 `audiovae_tile_frames=1` + 缩小 4 个 graph arena，是从"VoxCPM2 默认 OOM"应急出来的，本质是 **AudioVAE 解码器的显存配置**。但 VoxCPM2 的耗时构成是：

```
generator_ms              7780 ~ 7846   (91%)
audiovae_decoder_ms        766 ~  791   ( 9%)
```

我扫描了 6 组纯性能参数（tile=0/1/4、mem_saver on/off、full/small arena、latent_capacity），结果：

| 配置 | generator | VAE | RTF |
|---|---|---|---|
| base_tile1_small | 7846ms | 790ms | 0.378 |
| tile0 | 7780ms | 767ms | 0.375 |
| memsaver_off | — | — | 0.377 |
| tile1_fullarena | — | — | 0.376 |
| tile1_latentcap | — | — | 0.386 |
| tile4 | 7846ms | 768ms | 0.377 |

**全部 SHA256 一致**（`C360888D...`），但 RTF 差异 <3%。证明：调 VAE 参数对总耗时几乎无影响，我一直在错误的层面调参。6 步全部无损这点是对的，但优化方向错了。

---

## 三、真正的优化空间（按实测收益排序）

### 1. DiT estimator 的 H2D/D2H 往返（最大，未做）

`generator.cpp:773-785` 每一步做：

```cpp
ggml_backend_tensor_set(x_, ...)              // H2D
ggml_backend_tensor_set(cond_, ...)
ggml_backend_tensor_set(mu_, ...)
ggml_backend_tensor_set(time_embedding_, ...)
ggml_backend_tensor_set(delta_time_embedding_, ...)
compute_backend_graph(backend, graph_)        // kernel 提交
ggml_backend_synchronize(backend)             // 同步
ggml_backend_tensor_get(output_, ...)         // D2H
```

6 步 × 每步 5 次上传 + 1 次同步 + 1 次下载。CUDA Graph 只包住 `compute_backend_graph` 里的 kernel，包不住这些外层往返。

**优化方案**：把 x/cond/mu/time/dt 输入张量改成 device 常驻，用 device 侧 kernel 生成 time embedding，避免每步 CPU 算好再上传；把输出留在 device 供下一步用，只在最后读回。这需要改 `generator.cpp` 并重编译。

### 2. DiT estimator 的专属融合算子（大，未做）

DiT 里 `time + delta_time_mlp` 是固定的小 MLP，可以融合成一个 kernel；attention 的 QKV 也可以像 MiniCPM 那样 packing（`cuda_perf` 只做了 LM 没做 DiT）。audio.cpp 已有 `PackedLinearWeightsBuilder` 现成工具，可复用。

### 3. CMS（ Chair / classifier-free guidance）批处理（中，未做）

CFG 已是 batched forward（`x_in` 里 x 重复两遍），这块设计是对的，无浪费。

### 4. 重编译可行性

工具链在（VS2022 BuildTools，cl.exe 14.43，cmake 在 Common7/IDE/CommonExtensions/Microsoft/CMake），只是 PATH 不完整——和当初 VoxCPM.cpp 情况一样。audio.cpp 没有现成 build 目录，需要从零配置（约数十分钟编译）。

---

## 四、结论

1. 我之前的"针对性调优"确实是假的——只是把 VoxCPM 0.5 的 VAE 显存配置搬过来，对 VoxCPM2 的真实瓶颈（DiT generator 7.8s）无效。
2. VoxCPM2 有专属算子但只覆盖 LM 投影，DiT 没有；llama.cpp 最新 kernel 大部分因 Pascal 架构不匹配而无法移植。
3. **真正有空间的是 DiT 的 device 常驻输入 + 专属融合算子**，需要改 `generator.cpp` 并重编译。这是一项明确、可落地、有实测依据的工作，尚未执行。
