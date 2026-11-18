# VoxCPM2 特化算子深度调查与实测（源码级）

日期：2026-11-18
硬件：GTX 1070 8GB（Pascal sm_61）
引擎：audio.cpp 0.9.5 源码，自编译原生 sm_61（非 PTX JIT）
编译工具链：VS2022 BuildTools cl 14.44 / CUDA 12.6 / Ninja，缓存全在 F 盘

---

## 一、这一轮做了什么（真的动了源码）

1. **配起完整编译环境**：audio.cpp 官方 `scripts/build_windows.ps1`，锁 `-CudaArchitectures 61-real`（原生 sm_61，而非预编译版的 PTX 向下兼容），`-ModelSet custom -Models voxcpm2` 只编 VoxCPM2。ggml 版本 0.12.0。
2. **编译出可复现基线**：`BUILD_EXIT=0`，70MB exe。中途遇 nvcc -j3 并行死锁（卡在 convert.cu 44 分钟），改 -j1 补编解决。
3. **验证基线等价**：自编译基线输出与官方预编译版 SHA256 **完全相同**（`bc9db498...` 短文本 / `c360888d...` 100字），证明编译配置正确。
4. **给 DiT estimator 加四阶段累加计时埋点**（`generator.cpp`，环境变量 `VOXCPM2_DIT_PROFILE=1` 控制，默认关闭零开销），增量编译后实测。
5. **跑 profile + threads/mem_saver/graph 扫描**。

git 版本链（audio.cpp 工作区）：
```
3f6d93f baseline: 未修改源码 (tag: baseline-upstream)
7916fda profile: DiT estimator 四阶段计时埋点
```

---

## 二、profile 数据：瓶颈到底是什么

100 字文本，143 个 patch，每个 patch 6 步扩散（首步零初始化跳过，实际 5 次 DiT forward），共 **715 次 DiT forward**：

| 阶段 | 总耗时 | 占比 | 每次 forward |
|---|---|---|---|
| **DiT compute（GPU kernel）** | **4194.6 ms** | **95.9%** | 5.87 ms |
| upload（5×tensor_set H2D） | 139.5 ms | 3.2% | 0.20 ms |
| readback（tensor_get D2H） | 40.6 ms | 0.9% | 0.06 ms |
| sync（cuda synchronize） | 1.7 ms | 0.04% | 0.002 ms |

generator 总计 7846 ms，DiT compute 占其中 **53.5%**。

**结论**：我上一轮说的"CPU↔GPU 往返是瓶颈"是错的——sync 只有 1.7ms/全程。CUDA Graph 已把 kernel launch 和同步开销压到可忽略。真正的耗时是 12 层 Q8_0 matvec 的纯 GPU 计算。

---

## 三、各项优化的真实效果（全部实测，全部无损）

所有配置输出 SHA256 均为 `c360888d48cefd9b9619`，数值零改变。

| 优化 | 效果 | 结论 |
|---|---|---|
| `perf_mode=cuda_perf`（QKV/gate-up packing） | RTF 0.410→0.398（−2.9%） | 有效，已启用。且**已覆盖 DiT**（`minicpm.cpp:261` DiT decoder 也传了 optimized），不是只覆盖 LM |
| CUDA Graph（`GGML_CUDA_GRAPHS_PRE_AMPERE`） | DiT compute 4357→4195ms（−3.9%），RTF −1.4% | 有效但小，已启用 |
| threads 1/4/8/16 | DiT compute 4223–4231ms（差异<0.3%） | 无影响 |
| mem_saver on/off | 4223 vs 4233ms | 无影响 |
| audiovae tile 0/1/4 + arena/latent_capacity（上一轮） | RTF 全在 0.375–0.386 | 只影响 0.77s 的 VAE，对 7.8s generator 无效 |

---

## 四、llama.cpp 最新优化为什么用不上（逐项核实源码）

拉取最新 ggml（`_ggml_latest/cuda/`）与 audio.cpp 内嵌版（2026-11-17）逐文件对比：

| 新优化 | 触发条件（源码实证） | 本场景 |
|---|---|---|
| `halve_iters`（mmvq 1615 vs 1180 行） | `if (table_id != MMVQ_PARAMETERS_GB10) return false;` — Blackwell sm_100 专用表 | ❌ Pascal 用不上 |
| `small_k` | 两版 `should_use_small_k` 判定逻辑**完全相同** | ⚠️ 已存在 |
| `ncols_dst=2/3/4...` matvec | 两版代码**逐字相同** | ⚠️ 已存在 |
| sparse attention（fattn 838 vs 593） | 超长 context mask 压缩 | ❌ DiT 序列很短 |
| PDL | sm_90+ | ❌ Pascal 不支持 |
| MoE matvec | MUL_MAT_ID | ❌ 非 MoE 模型 |

**结论**：不是 llama.cpp 没优化，是这些新优化没有一个适用于 "Pascal sm_61 + batch=2 + 小序列 + 非MoE" 这个组合。盲目整文件替换会因 API 变化编译失败，且不会带来收益。

---

## 五、最终结论（这次有源码 + profile 双重依据）

- VoxCPM2 的专属算子 audio.cpp 已做到位：LM + DiT + encoder 的 QKV/gate-up packing（`cuda_perf`）、CUDA Graph、prompt cache、AudioVAE tiling 全都有。
- 剩余 4.2s 的 DiT compute 是 12 层 × hidden1024 × ffn4096 × batch2 的 Q8_0 matvec，在 Pascal（无 Tensor Core、仅 DP4A）上是硬件计算极限。
- 在不改数值、不降 6 步、Q8_0、Pascal 硬件的硬约束下，**已无可落地的算子级提速空间**。这是实测结论，不是拍脑袋。
- 若解除约束：换 Turing+ 显卡（Tensor Core/INT8 DP4A 更快）、或降步数/降精度（用户已明令禁止），才可能有显著提升。

---

## 六、可复现命令

```bash
# 编译（已配置好 build/windows-cuda-release）
cmake --build build/windows-cuda-release --target audiocpp_cli -j 1

# profile
set VOXCPM2_DIT_PROFILE=1
set GGML_CUDA_GRAPHS_PRE_AMPERE=1
audiocpp_cli --task tts --family voxcpm2 --model <voxcpm2-q8_0.gguf> \
  --backend cuda --text "..." --out out.wav --threads 8 --seed 1234 \
  --num-inference-steps 6 --metrics --log-file profile.log \
  --session-option perf_mode=cuda_perf --session-option audiovae_tile_frames=1
# 查看 voxcpm2.dit.{upload,compute,sync,readback}_ms 累加值
```
