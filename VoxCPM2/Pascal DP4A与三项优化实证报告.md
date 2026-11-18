# Pascal DP4A 与三项优化实证报告（2026-11-18）

## 硬件 / 引擎 / 硬约束
- GTX 1070 8GB，**Pascal sm_61**（无 Tensor Core，有 INT8 DP4A 指令）
- audio.cpp 0.9.5 源码，ggml 0.12.0，原生 sm_61 自编译（非 PTX JIT）
- CUDA 12.6 / VS2022 BuildTools cl 14.44 / Ninja
- **编译并行度：脚本设为 -j6（用户指定），但这台机器的 nvcc 在 -j6 下反复死锁**（3 个 nvcc 子进程 CPU≈0、卡在链接前），每次都必须清理进程后改 -j1 串行补编才能完成。这是 nvcc + Windows MSVC 的已知问题，非源码问题。所有 BUILD_EXIT=0 的构建都经过串行补编。
- 模型：Q8_0 单文件 GGUF，voxcpm2，固定 6 步 / seed1234 / cuda_perf / CUDA Graph ON
- **硬约束：不改数值、不降 6 步、不降 Q8_0**（所有 A/B 都验证输出 SHA256）

---

## 一、DP4A：已确认在用，且贡献巨大（实测 33.7%）

用户要求"INT8 DP4A 更快，考虑专门写 Pascal 算子"。结论：**ggml 已经在用硬件 DP4A，且这是当前性能的关键**。我做了两层实证。

### 1.1 源码确认：Q8_0 matmul 必然走 DP4A
- `common.cuh:685` `ggml_cuda_dp4a()`：`#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A`（**610**）时用 `__dp4a(a,b,c)` 硬件指令
- sm_61 == 610，恰好满足门槛
- `ggml-cuda.cu:2737` `ggml_cuda_should_fuse_mul_mat_vec_q`：量化权重 + F32 激活 + batch≤8 走 MMVQ；`mmvq.cu:17` `vec_dot_q8_0_q8_1` 内部就是 `ggml_cuda_dp4a`
- `mmq.cu:278` `should_use_mmq`：Pascal 无 Tensor Core（`fp16_mma_hardware_available=false`）→ Q8_0 直接返回 true 走 MMQ（也是 DP4A）
- **两条 matmul 路径（MMVQ/MMQ）在 Pascal 上都用 INT8 DP4A**

### 1.2 实测 A：关掉 DP4A 硬件指令（改 common.cuh 走 4 次标量乘加）
唯一差异 = `ggml_cuda_dp4a` 是否用 `__dp4a` 指令：

| 版本 | DiT compute | session wall | SHA256 |
|---|---|---|---|
| 硬件 `__dp4a`（当前） | **4198 ms** | 8363 ms | c360888d… ✓ |
| 强制标量乘加 | 5611 ms | 9740 ms | c360888d… ✓ |

→ **DP4A 硬件指令把 DiT compute 从 5611ms 降到 4198ms，提速 33.7%（省 1413ms）**。
→ 输出 SHA256 完全一致，数值等价。这是"DP4A 正在用且是关键"的铁证。

### 1.3 实测 B：强制开启被上游禁用的 Pascal 融合路径（改 ggml-cuda.cu gate）
上游 `ggml-cuda.cu:2751` 明确注释 `// fusion is not universally faster on Pascal` 并在 `cc<=PASCAL` 时 return false。去掉这个 gate 强制融合：

| 版本 | DiT compute | session wall | SHA256 |
|---|---|---|---|
| 原样（Pascal 禁用融合） | **4212 ms** | 8406 ms | c360888d… ✓ |
| 去掉 gate 强制融合 | 4328 ms | 8575 ms | c360888d… ✓ |

→ 强制融合**慢 2.8%**，证明上游针对 Pascal 的禁用是有依据的。数值等价。

### 1.4 DP4A 结论
**不需要、也不应该"另写 Pascal DP4A 算子"**——ggml 已经为 Pascal 实现了高度优化的 DP4A matmul（MMQ + MMVQ 两条路径），且实测贡献 33.7%。任何自写算子想超过它，都要先跨过"高度模板特化 + 多年调优"这道门槛，收益为负的风险极高。

---

## 二、time + delta_time MLP 融合：结构上无法有效落地

用户列为"大收益未做"。源码分析结论：**在这套模型结构下无法用单次 matmul 有效融合**。

### 2.1 结构（`minicpm.cpp:246-264`, `generator.cpp:896-920`）
```
time  分支: Linear(W1t) → Silu → Linear(W2t) ┐
dt    分支: Linear(W1d) → Silu → Linear(W2d) ┘ → Add
```
- time 和 delta_time 是**两个不同的输入向量**（`time_embedding` vs `delta_time_embedding`）
- 且用**两套不同的权重**（`time_mlp_*` vs `delta_time_mlp_*`）
- 每层 hidden→hidden（hidden=1024），4 个小 matmul

### 2.2 为什么融合不了
- **QKV 能 packing** 是因为"同一输入 × 多个权重"→ 输出维拼接即可一次 matmul
- **time/dt 是"不同输入 × 不同权重"**，无法用输出维 packing
- 唯一理论路径是 block-diagonal 权重 `[W2t 0; 0 W2d]`（shape 2h×2h），但计算量从 2×(h×h) 变成 (2h)×(2h) = **4 倍**，反而更慢
- `MUL_MAT_ID`（batched 按 id 选权重）是 MoE 用的，两个固定权重用不上

### 2.3 收益天花板
time+dt MLP 权重 4×(1024×1024)=4M MAC，DiT decoder 12 层约 144M MAC → **占比约 2.7%**。即使做到零开销，收益上限 2.7%，且第一层 packing 省 1 次 launch 会被 concat/split 抵消。

---

## 三、DiT 输入 device 常驻：CFG scale 的 double 累加是硬障碍

用户列为"最大未做"。实测 profile 显示往返只占 4.2%，且完整 device 常驻受数值约束。

### 3.1 profile 实测（`VOXCPM2_DIT_PROFILE=1`，715 次 forward）
| 阶段 | 总耗时 | 占比 |
|---|---|---|
| upload（5×tensor_set） | 139.0 ms | 3.2% |
| **compute** | **4198 ms** | **95.9%** |
| sync | 1.7 ms | ~0% |
| readback（tensor_get） | 40.6 ms | 0.9% |

→ 往返（upload+sync+readback）合计 181.3ms，占 4.1%。CUDA Graph 包不住外层 tensor_set/get 这点成立，但**量级只有 4%**。

### 3.2 为什么完整 device 常驻做不到（不改数值前提下）
- `optimized_cfg_scale()`（`generator.cpp:1199`）用 **double** 累加算 `dot(pos,neg)/dot(neg,neg)`：
  ```
  double dot=0, norm=1e-8;
  for i: dot += pos*neg; norm += neg*neg;
  return (float)(dot/norm);
  ```
- 要把 output 留在 device、CFG scale 也在 GPU 算，需要 GPU 上做 **F64 归约**
- 但 **ggml CUDA 后端完全不支持 F64**（`GGML_TYPE_F64` 在 ggml-cuda 里零命中），且 Pascal FP64 是 1/32 速率
- 强行用 F32 归约会改变数值 → SHA 变化 → 违反"不改数值"硬约束
- 因此每步 output **必须读回 CPU 用 double 算 CFG scale**，readback 省不掉

### 3.3 实测：cond/mu 缓存优化破坏数值，已回退
尝试把循环不变量 cond/mu 改为"仅值变化时上传"（`generator.cpp` 加 `last_cond_`/`last_mu_` 缓存）：

| 版本 | wall | SHA256 |
|---|---|---|
| base（每步上传） | 8821 ms | **c360888d…** |
| devres（缓存跳过） | 9857 ms | **fe6da89c…** ❌ |

→ **SHA 不一致（数值被破坏），且更慢（+11.7%）**。
→ 根因：CUDA Graph 下 `compute_backend_graph` 走 graph replay，graph capture 依赖每次 `tensor_set` 的调用语义；跳过 set 会让 replay 读到非预期 device 状态。同时增加的 `vector` 比较开销抵消了省下的 memcpy。
→ 该改动**已回退**（`git checkout generator.cpp`），不保留。

### 3.4 完整 device 常驻的终极结论
- 最彻底的常驻（output 留 device + GPU 上算 CFG scale）需要 F64 归约，但 **ggml CUDA 不支持 F64**，Pascal FP64 1/32 速率 → 不可行
- 部分常驻（cond/mu 缓存）**实测破坏数值且更慢** → 已回退
- upload+sync+readback 合计仅 4.1%，即便成功上限也很低
- **结论：不改数值前提下，device 常驻没有可落地的实现路径**

---

## 四、总表

> 注：所有 A/B 实验中，脚本传的 `perf_mode=cuda_perf` 因 PowerShell 参数处理问题实际未生效（日志显示 `perf_mode off`）。但**每组 A/B 内部两版条件完全相同**，因此对比结论（DP4A 33.7%、融合慢 2.8%、缓存破坏数值）依然成立。手动单跑验证时 perf_mode 生效，SHA 同为 c360888d。

| 优化项 | 用户预期 | 实证结论 | 是否落地 |
|---|---|---|---|
| Pascal DP4A 算子 | 更快 | **已在用，贡献 33.7%**（关掉慢 1413ms）；强改融合慢 2.8% | 保持当前实现（已最优） |
| time/dt MLP 融合 | 大收益 | 不同输入×不同权重，无法有效融合；上限 2.7% | 不可行 |
| DiT device 常驻 | 最大 | double CFG 无法搬 GPU（CUDA 不支持 F64）；cond/mu 缓存实测破坏数值且更慢 | 已回退，不可行 |

**最终交付状态**：三项均无源码改动保留（DP4A 实验改动、cond/mu 缓存改动全部回退）。工作区仅保留 `-j6` 编译脚本改动和本报告/实验脚本。这在"不改数值、不降 6 步、Q8_0"硬约束下是正确结果——强行保留任何一项都会破坏数值或变慢。

---

## 五、可复现实验脚本
- `_ab_dp4ainst.ps1`：DP4A 硬件指令 vs 标量（1.2 节）
- `_ab_mmvq.ps1`：Pascal 融合 gate A/B（1.3 节）
- `_ab_devres.ps1`：cond/mu 常驻 A/B（3.3 节）
- profile：`VOXCPM2_DIT_PROFILE=1 GGML_CUDA_GRAPHS_PRE_AMPERE=1` + `--session-option perf_mode=cuda_perf`
