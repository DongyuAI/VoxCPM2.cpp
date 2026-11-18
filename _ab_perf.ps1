# VoxCPM2 实测：audio.cpp 0.9.5 / Q8_0 单文件GGUF / 6步 / GTX1070 Pascal
# A/B: perf_mode=off (默认) vs cuda_perf (特化融合优化)
# 硬约束：num_inference_steps=6 固定，不降质量换速度；同一段100字文本；同一 seed
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$env:PATH = "$root\Bin\audiocpp;" + $env:PATH

if (-not (Test-Path $model)) { Write-Output "FATAL: model not found: $model"; exit 1 }

function Get-VramMiB {
    $o = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $o) { return [int]($o.Split("`n")[0].Trim()) }
    return -1
}

function Run-Vox($label, $perfMode) {
    Write-Output "`n########## VoxCPM2 $label (perf_mode=$perfMode, steps=6, seed=1234) ##########"
    $out  = "$root\TTS\VoxCPM2\out_$label.wav"
    $log  = "$root\Tsinglin_Voice_TMP_run_$label.log"
    if (Test-Path $log) { Remove-Item $log -Force }

    $vramBefore = Get-VramMiB
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --num-inference-steps 6 `
        --session-option "perf_mode=$perfMode" `
        --metrics --log-file $log 2>&1 |
        Select-String -Pattern "compute capability|Using backend|RTF|wall|audio duration|decode|CUDA graph|error|failed|sampl|voxcpm2|VoxCPM2|load"
    $sw.Stop()
    $vramAfter = Get-VramMiB
    $sz = (Get-Item $out -ErrorAction SilentlyContinue).Length
    Write-Output "---- $label wall=$($sw.ElapsedMilliseconds)ms out=$sz B vram=${vramBefore}->${vramAfter}MiB exit=$LASTEXITCODE ----"
    Write-Output "---- full log: $log ----"
}

Run-Vox "perfOFF" "off"
Run-Vox "perfON"  "cuda_perf"
Write-Output "`n对比 run 日志的 wall / audio duration / RTF。"
