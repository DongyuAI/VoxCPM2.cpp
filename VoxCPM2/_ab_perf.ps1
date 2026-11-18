# VoxCPM2 正式 A/B：audio.cpp 0.9.5 / Q8_0 / GTX1070 Pascal / 6步
# 硬约束：num_inference_steps=6、seed=1234、100字固定文本、不降精度
# 显存可用配置（已验证 SHA256 与默认完全等价）：audiovae_tile_frames=1 + 缩小 host graph arena
# 三组：
#   A_base     perf=off        graph=off
#   B_packed   perf=cuda_perf  graph=off   (QKV/gate-up 融合 matmul，dtype 不变)
#   C_packedgr perf=cuda_perf  graph=on    (Pascal CUDA Graph via GGML_CUDA_GRAPHS_PRE_AMPERE)
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH

if (-not (Test-Path $model)) { Write-Output "FATAL: model not found"; exit 1 }
Write-Output "MODEL: $((Get-Item $model).Length) bytes"

$common = @(
    "--session-option", "audiovae_tile_frames=1",
    "--session-option", "dit_graph_arena_mb=256",
    "--session-option", "lm_step_graph_arena_mb=256",
    "--session-option", "audiovae_graph_arena_mb=256",
    "--session-option", "audiovae_encoder_graph_arena_mb=256"
)

function Get-VramMiB {
    $o = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $o) { return [int]($o.Split("`n")[0].Trim()) }
    return -1
}

function Run-Vox($label, $perfMode, $enableGraph) {
    Write-Output "`n########## $label perf=$perfMode graph=$enableGraph steps=6 seed=1234 ##########"
    $out = "$root\TTS\VoxCPM2\out_$label.wav"
    $log = "$root\TTS\VoxCPM2\log_$label.txt"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $out) { Remove-Item $out -Force }

    Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
    if ($enableGraph) { $env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1" }

    $v0 = Get-VramMiB
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --num-inference-steps 6 --metrics --log-file $log `
        --session-option "perf_mode=$perfMode" @common 2>&1 |
        Select-String -Pattern "RTF|wall|audio duration|cudaMalloc|out of memory|failed|error|CUDA graph"
    $sw.Stop()
    $v1 = Get-VramMiB
    Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
    $sz = (Get-Item $out -ErrorAction SilentlyContinue).Length
    Write-Output "---- $label wall=$($sw.ElapsedMilliseconds)ms out=${sz}B vram=${v0}->${v1}MiB exit=$LASTEXITCODE ----"
}

Run-Vox "A_base"     "off"        $false
Run-Vox "B_packed"   "cuda_perf"  $false
Run-Vox "C_packedgr" "cuda_perf"  $true
Write-Output "`n详细: TTS\VoxCPM2\log_*.txt"
