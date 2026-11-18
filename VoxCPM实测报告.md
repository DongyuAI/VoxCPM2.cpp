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
