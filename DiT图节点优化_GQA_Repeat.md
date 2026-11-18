# VoxCPM2 DiT 图节点级优化报告（GQA KV Head Repeat）

日期：2026-11-18
硬件：GTX 1070 (Pascal sm_61, 8GB)
硬约束：**6 步 / Q8_0 / 不改数值 / wav SHA 逐字节一致 / 缓存全在 F 盘**

---

## 一、为什么调 kernel 参数只有 1% 收益

此前针对 Pascal MMQ/MMVQ 的 kernel 参数调整（mmq_y 64→32）实测只有 ~1% DiT 收益。
根因是**优化目标选错了**：一直在优化矩阵乘 kernel，但矩阵乘根本不是 DiT 图的主要开销。

加 env-gated 诊断探针（`VOXCPM2_DIT_PROFILE` 输出 `graph_nodes` 与算子直方图，
`GGML_CUDA_LOG_MATMUL` 输出各 matmul 形状）后，实测 DiT estimator 单图：

```
总节点 1029
  CONCAT    363   <- 35%
  RESHAPE   231
  MUL_MAT   115   <- 真正计算只占 11%
  PERMUTE    87
  VIEW       49
  MUL        37
  ADD        32
  CONT       28
  RMS_NORM   25
  ROPE       24
  UNARY      14
  SOFT_MAX   12
  SCALE      12
```

结论：**超过 70% 的节点是数据搬运/视图，MUL_MAT 只占 11%**。
在 CUDA Graph 下节点数本身就是开销（每节点约 5.7µs，含 kernel launch + 依赖调度），
所以真正该优化的是**节点数量**，而非单个 matmul kernel 的 DP4A 效率。

---

## 二、根因：363 个 CONCAT 的来源

定位到 `src/models/voxcpm2/minicpm_blocks.h` 的 `repeat_kv_heads()`：

```cpp
// 原实现：GQA 从 4 个 KV head 扩展到 16 个 query head
for (int64_t head = 0; head < kv_heads; ++head) {   // 4 个 KV head
  auto one = SliceModule({1, head, 1}).build(ctx, input);
  for (int64_t rep = 0; rep < repeats; ++rep)       // repeats = 16/4 = 4
    heads.push_back(one);
}
output = heads.front();
for (i = 1; i < heads.size(); ++i)                  // 16 个 slice → 15 次二元 concat
  output = ConcatModule({1}).build(ctx, output, heads[i]);
```

- 每层 attention 的 K、V 各调用一次 → 2 × 15 = 30 个 CONCAT/层
- 12 层 transformer → 360 个 CONCAT
- 加上 generator 里 mu/time/cond 的 3 个 CONCAT → **正好 363 个**，与探针实测完全吻合。

字节序是「每个 KV head 连续重复 4 次」：`[h0,h0,h0,h0, h1,h1,h1,h1, ...]`。

---

## 三、优化方案（SHA-safe）

用 ggml 原生 metadata 操作重建相同字节序，把 15 个 concat kernel 降到 1 个 repeat：

```cpp
// ggml ne 是逻辑 dims 的反序：逻辑 {B,H,S,D} -> ggml ne {D,S,H,B}
ggml_tensor * c = ggml_cont(ctx.ggml, t);                     // transpose 输出需连续化
ggml_tensor * r = ggml_reshape_4d(ctx.ggml, c, D*S, 1, H, B);
ggml_tensor * p = ggml_repeat_4d(ctx.ggml, r, D*S, repeats, H, B);
ggml_tensor * o = ggml_reshape_4d(ctx.ggml, p, D, S, H*repeats, B);
```

关键：reshape 到 `{D*S, 1, H, B}` 后沿 ne1 做 `1→4` repeat，内存序为 `for h: for r:`
= `[h0,h0,h0,h0, h1,h1,...]`，与原 concat 链逐字节一致。

### 字节序正确性双重验证

1. **numpy 穷举验证**：用唯一 id 填充 `[B=2,H=4,S=11,D=64]`，
   对比原 concat 与新 reshape+repeat 序列，`np.array_equal == True`。
2. **wav SHA 验证**：见下方 A/B，四组输出 SHA 全部一致。

> 注：`ggml_cont` 不能省——`ggml_reshape_4d` 硬性要求输入 contiguous，
> transpose 输出是非连续 view。这带来每层 K/V 各 1 个 cont（共 24 个），
> 但换来减少 360 个 concat，净收益显著。

保留 env 逃生舱 `VOXCPM2_GQA_CONCAT=1` 可切回原路径做 A/B / 回滚，默认关闭（即默认开启优化）。

---

## 四、A/B 实测（100字 / 6步 / Q8_0 / seed1234 / cuda_perf / graph ON）

同一 exe，唯一差异 = env 开关，正反两轮排除热身效应：

| 轮次 | 版本 | DiT compute | wav SHA256 |
|---|---|---|---|
| 1 | 原 concat | 4240.9 ms | `c360888d…` |
| 1 | **新 repeat** | **3428.9 ms** | `c360888d…` |
| 2 | **新 repeat** | **3421.6 ms** | `c360888d…` |
| 2 | 原 concat | 4262.2 ms | `c360888d…` |

**SHA 四组全部一致 → 数值零改变。**

节点数变化（探针实测）：

| 算子 | 原 | 新 |
|---|---|---|
| CONCAT | 363 | **3** |
| VIEW | 49 | 1 |
| REPEAT | 0 | 24 |
| CONT | 28 | 52 |
| RESHAPE | 231 | 279 |
| 总节点 | 1029 | **717** |

端到端指标：

| 指标 | 原 concat | 新 repeat | 变化 |
|---|---|---|---|
| RTF | 0.3697 | **0.3296** | **−10.9%** |
| wall_ms | 8459 | **7540** | −919 ms |
| DiT compute | 4240.9 ms | **3428.9 ms** | **−19.1%** |
| peak RSS | 3633 MB | 3632 MB | 不变 |
| 音频时长 | 22880 ms | 22880 ms | 一致 |

---

## 五、结论与版本

- 这是本轮**第一个 SHA-safe 且收益显著（DiT −19%）**的优化落地。
- 印证了此前的方向修正：DiT 瓶颈是**图节点数（数据搬运）**，不是 matmul kernel 的 DP4A 效率。
- audio.cpp 本地 commit：`bc98edd`（上游基线 `a6e32214e762`）。
- 相对最初 baseline（官方 CLI，RTF 0.39 / DiT compute 4299ms）累计收益：
  DiT compute 4299 → 3429ms（−20%），RTF 0.39 → 0.33（−15%）。

### 后续可评估方向（未做，需实测）

- PERMUTE 87 个节点：部分 permute 是纯 stride view，部分会触发 copy，可进一步定位。
- CONT 52 个：我引入的 24 个 cont 是 reshape 前置必需，暂不可省。
- 把 cont 提前到 transpose 之前，从源头合并，改动大、风险高，需谨慎评估。
