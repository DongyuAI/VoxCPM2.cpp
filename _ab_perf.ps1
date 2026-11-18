# VoxCPM2 实测矩阵：audio.cpp 0.9.5 / Q8_0 单文件GGUF / 6步 / GTX1070 Pascal
# 硬约束：num_inference_steps=6、seed=1234、同一段100字文本，不降质量换速度
# 三组：
#   A base      = perf_mode off（上游默认）
#   B packed    = perf_mode cuda_perf（QKV/gate-up 融合 matmul，不改dtype）
#   C packedgr  = cuda_perf + GGML_CUDA_GRAPHS_PRE_AMPERE（Pascal CUDA Graph）
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$env:PATH = "$root\Bin\audiocpp;" + $env:PATH

if (-not (Test-Path $model)) { Write-Output "FATAL: model not found: $model"; exit 1 }
Write-Output "MODEL: $((Get-Item $model).Length) bytes"

function Get-VramMiB {
    $o = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $o) { return [int]($o.Split("`n")[0].Trim()) }
    return -1
}

function Run-Vox($label, $perfMode, $enableGraph) {
    Write-Output "`n########## $label perf=$perfMode graph=$enableGraph steps=6 seed=1234 ##########"
    $out  = "$root\TTS\VoxCPM2\out_$label.wav"
    $log  = "$root\TTS\VoxCPM2\log_$label.txt"
    if (Test-Path $log) { Remove-Item $log -Force }

    Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
    if ($enableGraph) { $env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1" }

    $v0 = Get-VramMiB
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --num-inference-steps 6 `
        --session-option "perf_mode=$perfMode" `
        --metrics --log-file $log 2>&1 |
        Select-String -Pattern "compute capability|Using backend|RTF|wall|audio duration|error|failed|CUDA graph|packed"
    $sw.Stop()
    $v1 = Get-VramMiB
    Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
    $sz = (Get-Item $out -ErrorAction SilentlyContinue).Length
    Write-Output "---- $label wall=$($sw.ElapsedMilliseconds)ms out=${sz}B vram=${v0}->${v1}MiB exit=$LASTEXITCODE ----"
}

Run-Vox "A_base"     "off"        $false
Run-Vox "B_packed"   "cuda_perf"  $false
Run-Vox "C_packedgr" "cuda_perf"  $true
Write-Output "`n详细日志: TTS\VoxCPM2\log_*.txt"
