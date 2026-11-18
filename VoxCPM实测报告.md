# VoxCPM.cpp 实测报告（GTX 1070 / Pascal sm_61）

> 实测日期：2026-11-17　硬件：GTX 1070 8GB（compute capability 6.1）+ CUDA 12.6 + MSVC 19.44
> 项目：bluryar/VoxCPM.cpp（Apache-2.0），ggml 0.9.7，Windows 源码编译 CUDA 版
> 状态：**已编译成功并跑通推理**，`BUILD_EXIT=0`

---

## 一、结果速览

| 指标 | q8_0（audiovae-f16） | q4_k（audiovae-f16） |
|---|---|---|
| 模型大小 | 810 MB | 544 MB |
| 生成音频时长 | 5.12 s | 3.20 s |
| decode 速度 | ~38 it/s（0.026 s/it） | ~38 it/s（0.026 s/it） |
| AudioVAE encode | 0.925 s | 0.914 s |
| 模型推理 | 2.609 s | 1.768 s |
| AudioVAE decode | 0.616 s | 0.394 s |
| **总耗时** | **4.149 s** | **3.076 s** |
| **RTF（完整流水线）** | **0.810**（超实时） | **0.961**（接近实时） |
| RTF（不含 VAE encode） | 0.630 | 0.676 |
| 质量（ASR 回译） | “大家好,我現在正在大可奇奇體驗AI科技” ≈ 原文 | “大家好,我现在正在打可机器体验AI科技” 有错字且漏读 |

**结论：q8_0 是最佳选择——质量明显更好，且 RTF 0.81 已超实时。q4_k 省 40% 时间和显存，但量化损伤导致漏读（时长从 5.12s 缩到 3.2s），不建议用于正式合成。**

---

## 二、运行命令（可复现）

```bash
cd F:/Tsinglin_Voice/TTS/VoxCPM.cpp
export PATH="F:/Tsinglin_Voice/TTS/VoxCPM.cpp/build-cuda3/bin:$PATH"   # DLL 所在目录

./build-cuda3/examples/voxcpm_tts.exe \
  --model-path ./models/voxcpm-0.5b-q8_0-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out_q8.wav \
  --backend cuda --threads 8
```

输出日志确认 Pascal 后端：`Device 0: NVIDIA GeForce GTX 1070, compute capability 6.1` + `Using backend: cuda (CUDA0)`。

**注意**：`voxcpm_tts.exe` 在 `build-cuda3/examples/`，但 `ggml*.dll` 在 `build-cuda3/bin/`。必须把 `bin` 加入 PATH（或把 DLL 拷到 exe 同目录），否则 exit 127。

参考音频 `examples/tai_yi_xian_ren.wav` 需与 `--prompt-text` 对应（这是该音频的原文）。要换音色，换成你自己的参考音频 + 对应文本即可（零样本克隆）。

---

## 三、编译踩坑记录（Windows + MSVC 专有）

项目主要面向 Linux/GCC，Windows 源码编译需 4 处修改（均已完成，记录在 `CMakeLists.txt`）：

1. **GCC flags 不兼容 MSVC**：原 CMakeLists 硬编码 `-Wall -Wextra / -g -O0 / -O3`，MSVC 报 D8021。改为 `if(MSVC)` 分支，只在非 MSVC 用 GCC flags，MSVC 用 `/utf-8 /bigobj`。
2. **CUDA 架构**：`-DCMAKE_CUDA_ARCHITECTURES=61`（sm_61 = GTX1070）。CMake 自动识别为 61-real。
3. **MSVC 专属选项误传 nvcc**：`add_compile_options(/utf-8 /bigobj)` 若不加限定会传给 nvcc，导致 `nvcc fatal: A single input file is required`。改用生成器表达式 `$<$<COMPILE_LANGUAGE:CXX>:...>` 限定只给 C/CXX。
4. **C++ 标准**：源码用了 designated initializers（C++20 特性），项目原设 C++17，MSVC 报 C7555。改为 `CMAKE_CXX_STANDARD 20`。
5. **M_PI_2 未定义**：`src/unified_cfm.cpp` 用 `M_PI_2`，MSVC 需 `_USE_MATH_DEFINES`。已全局 `add_compile_definitions(_USE_MATH_DEFINES)`。
6. **nlohmann_json 依赖**：原走 FetchContent（GitHub release，本网络 502）。已手动下载 3.11.3 放到 `third_party/json/`。
7. **Windows nvcc 并行死锁**：`-j 4` 会卡死（多个 nvcc CPU 归零），必须用 `-j 1` 串行。

构建脚本：`_build_cuda.ps1`（自包含 MSVC/SDK 环境设置）。产物在 `build-cuda3/`。

---

## 四、速度评估（针对“确保速度”）

- **RTF 0.81 意味着生成比播放还快**——5.12 秒音频只需 4.15 秒生成，足以支撑实时播放（首句需先生成约 1 秒缓冲）。
- decode 稳态 38 it/s，与量化位数关系不大（q8/q4 都是 38），说明 0.5B 规模下瓶颈是 codec/kernel 调度而非矩阵乘精度。
- **优化方向**（若要更低延迟）：
  - AudioVAE encode 占 0.925s，是固定开销。若做**参考音频预编码缓存**（离线算好 latent 复用），可省掉这部分，RTF 从 0.81 降到约 0.63。
  - 缩短文本 / 降 `--inference-timesteps`（默认 10）可进一步提速，但可能影响质量。
  - 流式路径（`--stream`）可做句级边生成边播放，降低首包延迟。

---

## 五、产物位置

```
F:/Tsinglin_Voice/TTS/VoxCPM.cpp/
├── build-cuda3/examples/voxcpm_tts.exe      # 主程序（+ server/quantize/imatrix）
├── build-cuda3/bin/ggml-cuda.dll             # CUDA 后端（84MB，sm_61）
├── models/voxcpm-0.5b-q8_0-audiovae-f16.gguf  # 推荐模型 810MB
├── models/voxcpm-0.5b-q4_k-audiovae-f16.gguf  # 对比模型 544MB
├── out_q8.wav / out_q4.wav                   # 实测生成样本（供试听）
└── _build_cuda.ps1                            # 可复现构建脚本
```

> 质量最终以试听为准；此处 ASR 回译是客观辅助指标。建议重点试听 `out_q8.wav`。

---

## 附录：CUDA Graph 移植尝试（结论：失败，不采用）

**背景**：尝试把 Pascal 上被上游 AMPERE 门槛挡住的 CUDA Graph 放行（ggml-cuda.cu 的 `ggml_cuda_graph_set_enabled` 把门槛从 AMPERE 降到 PASCAL），并开 `GGML_CUDA_GRAPHS=ON` 重新编译整个 ggml-cuda。编译成功（sm_61，66/66）。

**A/B 方法**：同一 q8_0 模型、同一文本、同一 6 步；用 `VOXCPM_BENCHMARK_DECODE_STEPS=40 + IGNORE_STOP=1` 固定两边都跑 40 步，排除随机 stop 时机干扰。

**结果（干净对比）**：
| 指标 | graphOFF | graphON |
|---|---|---|
| 稳态 decode | 43.51 it/s | 41.24 it/s（**慢 5%**） |
| Model inference(40步) | 1.634s | 1.679s |
| 首步 warmup | 无 | 20 it/s（capture 开销） |

**结论**：不采用。原因分析：
1. VoxCPM 上游已自带 CPU 侧 graph 缓存（`ensure_unified_cfm_graph` / `ensure_decode_front_half_graph`），已省掉 graph build 开销；CUDA Graph 是另一层，收益重叠。
2. 每步 kernel 较大（front_half ~7.5ms、base_step ~10ms），kernel launch 开销占比 <1%，graph 削减 launch 几乎无收益。
3. ggml 的 `ggml_backend_cuda_graph_compute` 主要对单 CUDA backend 生效，而 VoxCPM 走 scheduler 多 backend 路径，replay 收益被抵消。
4. Pascal 上 graph capture/replay 本身有额外开销。

> 注意：默认 baseline（build-cuda3，graph OFF）才是正式可用版本。build-cudagraph 仅作实验记录。

---

## 附录：提速路径二 —— 参考音频 latent 预编码缓存（结论：有效，采用）

**原理**：CLI 的 `voxcpm_tts.exe` 每轮都现场跑 `extract_prompt_features`（AudioVAE encode，固定 ~0.9s）。server 模式用 `VoiceStore`：注册音色时把 `prompt_feat` 编码一次存成 `prompt_feat.bin`，合成时直接读缓存，不重复 encode。**完全无损**（同一份 prompt_feat）。

**实测（都 q8_0 / 6 步 / 同一段文本）**：
| 模式 | 端到端 wall | AudioVAE encode |
|---|---|---|
| CLI 现场编码（3 次） | 4.43 / 4.96 / 4.43 s | 0.947 / 0.889 / 0.901 s |
| Server 缓存音色（3 次） | 2.08 / 2.28 / 2.44 s | 0（读缓存） |

**收益**：端到端快约 **1 倍**，省掉每轮 ~0.9s 的固定 encode 开销。

**正确用法**：
```
# 1) 启动 server（质量底线：--inference-timesteps 6）
voxcpm-server.exe --model-path <q8_0.gguf> --model-name voxcpm --voice-dir <dir> \
   --backend cuda --inference-timesteps 6 --disable-auth --port 8080

# 2) 注册音色（一次性 encode，落盘 prompt_feat.bin）
curl -X POST localhost:8080/v1/voices -F id=taiyi -F text="参考音频原文" -F audio=@ref.wav

# 3) 之后每次合成直接复用缓存
curl -X POST localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
   -d '{"model":"voxcpm","input":"要合成的文本","voice":"taiyi","response_format":"wav"}'
```

> 这就是长文强调的"不要每轮重算参考编码"。客服固定品牌音色场景下，注册一次、长期复用。

---

## 附录：提速路径三 —— 融合算子 / CPU 往返排查（结论：上游已做，空间极小）

**front_half 融合度**（`VOXCPM_PROFILE_FRONT_HALF=1`）：
```
lm_to_dit_proj:  8.477 ms
res_to_dit_proj: 4.034 ms
unified_cfm:     7.307 ms
split_total:    19.818 ms
fused_front_half: 9.429 ms   ← 融合版
fused_vs_split_gap: -10.389 ms  ← 已省一半
```
decode 三大阶段（正常路径，稳态合计 ~21ms/step）：front_half ~7.5ms + base_step ~10ms + residual_step ~3.5ms。

**CPU↔GPU 往返**（`VOXCPM_LOG_DECODE_TRANSFERS=1`）：
每步 h2d/d2h 均 0.000–0.009 MiB、0.03–0.36ms，占单步 <2%。无 `tensor_get→vector→tensor_set` 的 PCIe 往返浪费，无 CPU fallback。

**结论**：VoxCPM 上游已经完成最有价值的优化——front_half 融合（省 10ms/步）、全程 GPU 常驻、状态化 graph 缓存、预计算 CFM time table。Pascal 上再手写专用融合算子的收益空间很小且风险高，**不作为提速手段**。

> 注：开启 `VOXCPM_PROFILE_FRONT_HALF` 会因额外跑 split 对照版本把 decode 拖到 ~28 it/s，正常路径是 ~43 it/s，别用 profile 模式的数据当性能。

---

## 100 字合成：显存与耗时实测（GTX 1070 / q8_0 / 6 步 / server 缓存音色）

测试文本：101 汉字（"大家好，欢迎收听今天的节目……创造更好的明天。"），voice=taiyi（缓存），后端 CUDA，threads=8。

| 指标 | 实测值 |
|---|---|
| GPU 空闲基线 | 1487 MiB |
| 模型加载后常驻显存（空闲） | **2679 MiB** |
| 合成期间显存（3 次即时采样） | 3257 / 3267 / 3342 MiB（**峰值约 3.3 GB**） |
| 100 字端到端耗时（3 次） | 5.64 / 5.23 / 5.43 s（**稳定约 5.4 s**） |
| 输出音频时长 | 10.24 s（16kHz 单声道，327724 B） |
| RTF | **0.527**（合成比播放快约 1 倍） |

结论：8GB 的 GTX 1070 上，q8_0 + 6 步 + 缓存音色，100 字约 5.4 秒合成、峰值显存约 3.3GB，留有余量。常驻 2.7GB 意味着这张卡理论上还能再挂一路 worker（显存够，但算力并发会互相拖慢）。

样本：`试听样本/VoxCPM_100字_q8_6步.wav`（可试听质量）。
