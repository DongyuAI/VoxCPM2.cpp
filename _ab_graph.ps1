# VoxCPM CUDA Graph A/B 对比脚本
# 目的：在完全相同的条件下对比「Graph 关闭」vs「Graph 开启」的速度
# 硬约束：不改任何质量参数 —— 模型同为 q8_0，timesteps 显式锁定 6（质量底线），文本相同
# 用法：在 VoxCPM.cpp 根目录执行  powershell -ExecutionPolicy Bypass -File _ab_graph.ps1

$ErrorActionPreference = "Continue"
$root = "F:\Tsinglin_Voice\TTS\VoxCPM.cpp"
# A/B 两个可执行文件（每个用自己构建目录的 DLL，避免加载到错误的 ggml-cuda.dll）
$binOff = "$root\build-cuda3\bin"
$binOn  = "$root\build-cudagraph\bin"
$exeOff = "$root\build-cuda3\examples\voxcpm_tts.exe"      # 原版（GGML_CUDA_GRAPHS=OFF）
$exeOn  = "$root\build-cudagraph\examples\voxcpm_tts.exe"  # graph 版（PATCH + ON）

$model   = "$root\models\voxcpm-0.5b-q8_0-audiovae-f16.gguf"
$promptA = "$root\examples\tai_yi_xian_ren.wav"
$promptT = "对，这就是我，万人敬仰的太乙真人。"
$text    = "大家好，我现在正在大可奇奇体验AI科技。"
$steps   = "6"   # 质量底线，锁定；两边完全一致

$env:VOXCPM_LOG_DECODE_TIMING = "1"

function Run-One($label, $exe, $bin) {
    Write-Output "`n########## $label ##########"
    if (-not (Test-Path $exe)) { Write-Output "SKIP: 找不到 $exe"; return }
    # 把本构建的 bin 放到 PATH 最前，确保加载自己的 ggml-cuda.dll
    $env:PATH = "$bin;" + $env:PATH
    $out = "$root\ab_${label}_ts$steps.wav"
    & $exe `
        --model-path $model `
        --prompt-audio $promptA `
        --prompt-text $promptT `
        --text $text `
        --output $out `
        --backend cuda `
        --inference-timesteps $steps `
        --threads 8 2>&1 | Select-String -Pattern `
        "compute capability|Using backend|AudioVAE|Model inference|Total:|RTF|decode\]|Saved audio|Running prefill|graph"
    Write-Output "---- $label exit=$LASTEXITCODE  out=$out ----"
}

Run-One "graphOFF" $exeOff $binOff
Run-One "graphON"  $exeOn  $binOn
Write-Output "`n完成。对比两份 ab_graph*_ts6.wav 的 RTF 与 decode it/s；音频内容应完全一致（同一模型/同一 6 步）。"
