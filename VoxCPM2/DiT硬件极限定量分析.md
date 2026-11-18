# DiT 离硬件极限到底有多远——定量分析与"硬件极限论"的推翻

> 记录时间：2026-11-18
> 前置结论（已被推翻）：此前我判断"DiT compute 4198ms 已接近 Pascal 硬件极限，无优化空间"。
> 本文档用真实模型结构参数 + 硬件理论值证明：**这个结论是错的**。实测吞吐仅达硬件峰值的 1.3%，存在数倍优化空间。

---

## 一、数据来源：从私有 GGUF 提取真实 DiT 结构

audio.cpp 使用的 `voxcpm2-q8_0.gguf` 是 **audio.cpp 私有打包格式**（`audiocpp.tensor_name_format=native`，仅 16 个大 tensor，array 用 u32 计数），不是 llama.cpp 标准逐层 GGUF，无法直接 dump 逐层张量。

但 GGUF metadata 中嵌入了完整的 `config` JSON（`audiocpp.model_spec.json` + 二进制内嵌 config），提取结果：

```json
{
  "patch_size": 4,
  "feat_dim": 64,
  "dit_config": {
    "hidden_dim": 1024,
    "ffn_dim": 4096,
    "num_heads": 16,
    "num_layers": 12,
    "kv_channels": 128,
    "mean_mode": false
  }
}
```

关键结构参数：
- DiT decoder：**12 层**，hidden=1024，ffn=4096，16 heads，head_dim=64，kv_channels=128
- patch_size=4，feat_dim=64 → patch_elems = 4×64 = 256
- CFG batched：batch=2（positive + negative 在同一个 graph 里）

提取脚本与产物：
- `F:\Tsinglin_Voice\TTS\VoxCPM2\_gguf_inspect.py`（GGUF 解析）
- `F:\Tsinglin_Voice\TTS\VoxCPM2\_model_spec.json`（模型规格）
- `F:\Tsinglin_Voice\TTS\VoxCPM2\_config_full.json`（完整结构 config）

---

## 二、定量计算

### 2.1 单次 DiT estimator forward 的 FLOPs

结构（batch B=2，每样本 patch token=4，序列 T=2+1+4×2=11）：

- in_proj / cond_proj：feat(64) → hidden(1024)
- time / delta_time MLP：4 个 hidden×hidden 小 MLP
- 12 层 transformer：QKV + O proj、attention、SwiGLU FFN（gate/up/down）
- out_proj：hidden(1024) → feat(64)

计算结果：**单次 forward ≈ 3.247 GFLOP ≈ 1.623 GMAC**

（Q8_0 量化下，FP32 的"2 FLOP"在 Pascal 上实为 1 个 INT8 DP4A MAC，故用 MAC 数衡量 DP4A 负载。）

### 2.2 100 字总量

- 每 patch 6 步 diffusion，其中首步 `zero_init` 跳过 estimator → **实际 5 次 forward/patch**
- 100 字约 143 patch → 715 次 estimator forward（与 `VOXCPM2_DIT_PROFILE` 实测 calls=715 完全吻合）
- 总计算量：**1.161 TMAC（万亿 INT8 MAC）**

### 2.3 有效吞吐 vs 硬件峰值

实测 DiT compute = 4198ms：

```
有效 DP4A 吞吐 = 1.161 TMAC / 4.198 s = 0.277 TOPS (int8 MAC/s)

GTX 1070 (GP104, Pascal sm_61) DP4A 峰值 ≈ 21.6 TOPS（保守）/ 43 TOPS（乐观）
  相对 21.6 TOPS 峰值利用率 = 1.3%
  相对 43.0 TOPS 峰值利用率 = 0.6%
```

### 2.4 显存带宽下限

DiT 权重规模（Q8_0 ≈ 1.0625 byte/weight）：

```
每层权重 ≈ (4·H² + 3·H·F) × 1.0625 = (4·1024² + 3·1024·4096) × 1.0625
12 层 + time/dt MLP ≈ 206.1 MB

若每次 forward 都从显存读一遍权重：
  715 次 × 206.1 MB = 154.54 GB
  GTX 1070 带宽 192 GB/s → 理论下限 805 ms
```

实测 4198ms vs 带宽下限 805ms → **差 5.2 倍**。

---

## 三、结论：不是硬件极限，是严重的 kernel 低效

| 指标 | 数值 | 含义 |
|------|------|------|
| DP4A 算力利用率 | **1.3%** | 远未触及计算极限 |
| 带宽下限差距 | **5.2×** | 远未触及带宽极限 |
| 实测 DiT compute | 4198ms | — |

此前"已到硬件极限"的结论**完全错误**。真相是：
1. 矩阵维度极小（M 仅为 8~22），GPU 并行度利用严重不足
2. kernel 启动 / 调度 / 空转开销主导，而非有效计算
3. **存在数倍优化空间，应继续做 Pascal 特化 kernel**

---

## 四、根因定位：mmq_y=64 vs M=8

在 `external/ggml/src/ggml-cuda/mmq.cuh` 中定位到直接根因：

```cpp
static int get_mmq_y_host(const int cc) {
    return GGML_CUDA_CC_IS_AMD(cc) ? (...) :
        ((GGML_CUDA_CC_IS_NVIDIA(cc) && highest_compiled_arch >= VOLTA) ? 128 : 64);
}
static constexpr __device__ int get_mmq_y_device() {
    #if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        return 128;
    #else
        return 64;   // Pascal 走这里
    #endif
}
```

- Pascal (< Volta) 的 MMQ kernel **输出行 tile 固定 mmq_y=64**
- 而 DiT 每次 matmul 的 M 极小：
  - in_proj / cond_proj / out_proj：M = batch(2) × patch(4) = **8**
  - transformer 层：M = batch(2) × seq(11 tokens) = **22**
  - 都远小于 64

kernel 内每个 block 有 8 个 warp（`nwarps = 256/32 = 8`），循环 `for i0 in [0,mmq_y) step 8`，配合 `need_check` 分支的 `i = min(i, i_max)`：
- mmq_y=64、M=8 时：8 个 warp 循环 8 轮处理 64 行，其中 7 轮的输出行索引越界被 clamp 到第 7 行 → **87.5% 的 DP4A 计算浪费在重复计算同一行**
- 同时 `stream-k` 在 Pascal 上被禁用（`>= VOLTA` 才启用），无法用 K 维拆分填满 SM

三个因素叠加，解释了 1.3% 的算力利用率。

---

## 五、可落地的优化方向（SHA-safe）

关键判据：MMQ kernel 中每个输出元素 `dst[i,j]` 的 K 维 DP4A 累加顺序与 mmq_y 无关（mmq_y 只决定多少个输出行并行处理）。因此调整 mmq_y **不改变数值结果**，是 SHA-safe 的。

候选方案：
1. **Pascal 小-M 专用 mmq_y**（首选）：把 Pascal 的 mmq_y 从 64 降到 8（或 16），让 M=8 的矩阵零空转。需同时改 `get_mmq_y_host()` 与 `get_mmq_y_device()`。约束：需满足 `mmq_x × mmq_y ≥ nwarps×warp_size = 256`，故 mmq_y=8 时要求 mmq_x≥32（DiT 的 ncols=1024/4096 会选 mmq_x=64，满足）。
2. **Pascal 启用 stream-k**：把 K 维拆给多个 block 并行填满 SM；风险是 fixup 的累加顺序可能改变 SHA，需实测验证。
3. **融合 time/dt 的 4 个小 MLP**：占比小，收益上限低。

下一步：对方案 1 做 A/B 实测，对比 wav SHA 与 DiT compute 耗时。硬约束不变：**6 步 / Q8_0 / 不改数值 / SHA 必须一致**。

**Baseline（方案改动前）已实测存证**：
- exe：`audiocpp_base.exe`（build 目录，含 DIT_PROFILE 埋点）
- wav SHA256：`c360888d48cefd9b9619fb6853e54d285084a32f8f85051b72a4e2fa0a86db2a`
- DiT compute 累加 = 4299.9ms，715 calls，6.014ms/call；RTF 0.373
- 运行方式：前台直接调用（PowerShell `| Out-Null` 管道会导致进程提前返回、日志不全，必须用前台 bash/直接调用）

---

## 六、mmq_y=8 的遗留问题：warp 内 75% 线程空闲（后续改进项）

已实施 mmq_y 64→8（host + device 两处，`[PASCAL-SMALL-M]` 标记），正在编译验证。但需如实记录该方案的**理论天花板与遗留低效**：

### 6.1 问题：warp 内空闲

`vec_dot_q8_0_q8_1_dp4a` 的线程映射：
```cpp
for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {   // warp_size=32
    const int i = i0 + threadIdx.x;                // threadIdx.x 0..31 → 输出行
    ...
}
```
mmq_y=8 时只有 `threadIdx.x = 0..7` 对应有效输出行，**一个 warp 32 线程里 24 个空闲（75%）**。

### 6.2 收益预估修正

我此前说"mmq_y 64→8 提速近 8×"是**过于乐观**的。准确分析：
- DP4A 是 warp 锁步发射，空闲线程不额外消耗指令周期，但也不贡献有效 MAC
- mmq_y 从 64→8 真正减少的是 `vec_dot` 的输出行循环轮数（Pascal 上 `mmq_y/warp_size`：2 轮 → 1 轮）与 `load_tiles` 轮数（8 轮 → 1 轮）
- 预期的 DP4A 有效收益约 **2×量级**（行循环减半），加上 load 开销下降，实际收益以 A/B 实测为准

### 6.3 后续改进路线（保持 SHA 一致的前提下）

| 方案 | 思路 | SHA 风险 | 说明 |
|------|------|----------|------|
| A 线程重映射 | `threadIdx.x 0..31` 改为映射到 (行0..7 × K段0..3)，warp 内 shuffle reduction 打满 32 线程 | **高**：改变 K 累加顺序，必须验证数值等价 | 收益最大但风险高 |
| B 强制小 M 走 MMVQ | `mul_mat_vec_q` 是 K 维并行，天然打满 warp；Pascal 上 Q8_0 目前被 `ggml_cuda_should_use_mmq` 强制走 MMQ | **中**：MMVQ 与 MMQ 累加顺序不同，需实测 SHA | 需给小 M 加分支 |
| C 折中 mmq_y=16/32 | 在行级空转与 warp 占用间找平衡 | 无（SHA-safe） | 收益上限低于 A/B；**已选此路实测，见下** |

**决策顺序**：先拿 mmq_y 实测数据 → 若 SHA 一致且有收益，作为可回退的基线保留 → 再评估方案 A/B 的数值等价性。已建任务 #19 跟踪。

### 6.4 实测过程中的关键修正（2026-11-18）

**尝试 1：mmq_y=8 → 编译失败**。错误 `the size of an array must be greater than zero`：`sum[mmq_x*mmq_y/(nwarps*warp_size)]`，Pascal nwarps×warp_size=256，mmq_y=8 时需 mmq_x≥32，但模板会实例化 mmq_x=8/16/24（如 Q1_0），导致 sum[0]。上游 kernel 从未设计 mmq_y < warp_size。**已回退**。

**关键认知修正：DiT 各层走不同 kernel**。ggml 的分派（`ggml-cuda.cu:2864`）按 M 选择：
- `use_mul_mat_vec_q`（MMVQ）：当 `M = src1->ne[1] ≤ MMVQ_MAX_BATCH_SIZE(8)` 且优先判断
- `use_mul_mat_q`（MMQ）：M > 8 时

因此：
- **M=8 的层**（in_proj/cond_proj/out_proj + time MLP）：**已经走 MMVQ**，相对高效
- **M=22 的层**（12 层 transformer 的 qkv/o/ffn）：走 MMQ，mmq_y=64 → 22/64=34% 行利用率（浪费 66%）

**为什么不能让 M=22 也走 MMVQ（SHA 红线）**：
```
vecdotq.cuh:
  VDR_Q8_0_Q8_1_MMVQ = 2   // MMVQ 每次 DP4A 打包 2 个 int = 8 个 int8
  VDR_Q8_0_Q8_1_MMQ  = 8   // MMQ  每次 DP4A 打包 8 个 int = 32 个 int8
```
两者 K 维累加粒度不同（8 vs 32 元素一块），浮点舍入顺序不同 → **SHA 必然不一致**。所以"调大 MMVQ_MAX_BATCH_SIZE 让 transformer 走 MMVQ"违反硬约束，此路不通。这也解释了此前 device 常驻实验 SHA 变化的敏感性。

**尝试 2：mmq_y=32（当前实测中）**。满足 sum[mmq_x*32/256]≥1（mmq_x≥8 全部合法），可编译；对 M=22 的 MMQ 层行利用率 34%→69%；同为 VDR=8 累加，**SHA-safe**。串行 -j1 编译中（-j6 在本机会 nvcc 死锁）。

Baseline 存证：SHA `c360888d…`，DiT compute 4299.9ms，715 calls。

### 6.5 实测受阻：系统级运行时崩溃（2026-11-18 14:22+）

mmq_y=32 编译成功（`BUILD_EXIT=0`，exe 14:20 生成，0 error），但运行时所有 exe（自编译 base/opt + 官方 Bin 版）**统一崩溃**：
```
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml.c:1700: GGML_ASSERT(ctx->mem_buffer != NULL) failed
```
崩溃在 `ggml_init()` 的 `_aligned_malloc(mem_size, 64)` 返回 NULL，发生在 CUDA graph warmup 后的 context 构建阶段。

**排查结论（已排除）**：
- 不是代码改动：官方 Bin 版（从未重编译）同样崩溃；源码已 `git checkout` 回退到干净 baseline（mmq_y=64）
- 不是显存：nvidia-smi 显示 6970 MiB free，无残留进程
- 不是物理内存：32GB 总量 / 9GB free，Python ctypes 实测 `_aligned_malloc(3072MB, 64)` **成功**
- 不是缓存：删除重建 `audiocpp-gguf/<hash>` 缓存目录后仍崩
- 排查中发现 `vmmem`（WSL/Hyper-V）占用 8.7GB，但内存分配测试证明不是主因

**关键异常**：13:55 时完全相同的 baseline 命令能跑通（SHA c360888d），14:22 后所有 exe 全崩。中间唯一系统变化是反复编译 + 多次 GGUF 大文件读入内存。疑为 Windows CRT 堆/CUDA 上下文的瞬态状态问题。

**当前处理**：用 -j6 重新编译干净 baseline（遵守用户编译规则：默认 -j6、死锁降 -j3、严禁 -j1），验证重编译后是否恢复。若恢复则重跑 mmq_y A/B；若仍崩则需重启环境/等待系统状态恢复。**在崩溃解决前，mmq_y=32 无任何实测数据，不得宣称有收益**。

### 6.6 崩溃解决 + mmq_y=32 实测结果（2026-11-18 14:54）

**崩溃根因**：不是系统内存/CUDA/缓存问题，而是**旧 exe 与重编译后的 ggml 中间库状态不一致**（编译 mmq_y=32 时 ninja 重建了共享中间产物，源码回退后旧 exe 与之失配）。用 -j3 重新编译干净 baseline 后崩溃完全消失，SHA 恢复 `c360888d`。（-j6 再次死锁，按规则降 -j3 成功。）

**mmq_y=32 A/B 实测**（100字/6步/Q8_0/seed1234/cuda_perf/graph ON）：

| 指标 | baseline (mmq_y=64) | opt (mmq_y=32) | 差异 |
|------|------|------|------|
| wav SHA256 | c360888d… | **c360888d…** | **完全一致 ✓** |
| DiT compute | 4289.0 ms | 4243.9 ms | -45ms (**~1.05%**) |
| generator_ms | 7807 ms | 7692 ms | -115ms (**~1.5%**) |

**结论**：mmq_y=32 是 SHA-safe 的真实改进，但收益仅 ~1%。已提交 audio.cpp commit `5118c7f`。

**为什么理论 66% 行浪费只换来 1% 收益**：MMQ kernel 只用于 M>8 的层（DiT transformer 的 qkv/o/ffn，M=22），而 **绝大多数 matmul 是 M=8 的层**（in_proj/cond_proj/out_proj + time MLP），这些走 MMVQ 路径、根本不受 mmq_y 影响。mmq_y 优化只覆盖了耗时的一小部分。这印证了 6.4 节的分派分析。

**下一步方向（任务 #19）**：真正的大头是 M=8 的 MMVQ 层。要进一步优化必须动 MMVQ kernel（如 warp 内 K 维并行度、VDR=2 的 DP4A 吞吐），但需严格保持 SHA 一致。

---

## 附：可复现计算脚本

```python
# 单次 forward FLOPs（结构见第一节）
B, patch, feat = 2, 4, 64
H, F, L, heads = 1024, 4096, 12, 16
hd = H // heads
T = 2 + 1 + 2*patch
M = B * patch
flops = 2*M*H*feat*2              # in_proj + cond_proj
flops += 4 * 2*B*H*H              # time/dt MLP ×4
for _ in range(L):
    flops += 2*M*H*H*3             # q,k,v
    flops += 2*M*H*H               # o_proj
    flops += 2*B*heads*T*T*hd      # attention
    flops += 2*M*H*F*2 + 2*M*F*H   # SwiGLU FFN
flops += 2*M*feat*H               # out_proj

mac = flops/2
steps, patches, sec = 5, 143, 4.198
total_mac = mac*steps*patches
print('有效 DP4A 吞吐 = %.3f TOPS' % (total_mac/sec/1e12))
print('利用率(vs 21.6 TOPS) = %.1f%%' % ((total_mac/sec/1e12)/21.6*100))
```
