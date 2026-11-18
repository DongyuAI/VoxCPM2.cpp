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

下一步：对方案 1 做 A/B 实测（baseline exe 已备份为 `audiocpp_cli_baseline_streamk.exe`），对比 wav SHA 与 DiT compute 耗时。硬约束不变：**6 步 / Q8_0 / 不改数值 / SHA 必须一致**。

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
