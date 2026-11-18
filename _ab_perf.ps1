# VoxCPM2 实测：q8_0 / 6步 / GTX1070 Pascal
# A/B: perf_mode=off (默认) vs cuda_perf (特化融合优化)
# 全部锁 num_inference_steps=6，同一段100字文本，不改质量参数
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models"
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$env:PATH = "$root\Bin\audiocpp;" + $env:PATH

function Run-Vox($label, $perfMode) {
    Write-Output "`n########## VoxCPM2 $label (perf_mode=$perfMode, 6步) ##########"
    $out  = "$root\TTS\VoxCPM2\out_$label.wav"
    $log  = "$root\TTS\VoxCPM2\run_$label.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 `
        --request-option num_inference_steps=6 `
        --session-option "perf_mode=$perfMode" `
        --metrics --log 2>&1 | Tee-Object -FilePath $log |
        Select-String -Pattern "compute capability|Using backend|RTF|wall|audio duration|decode|CUDA graph|error|failed|sampl"
    $sw.Stop()
    $sz = (Get-Item $out -ErrorAction SilentlyContinue).Length
    Write-Output "---- $label wall=$($sw.ElapsedMilliseconds)ms size=$szB exit=$LASTEXITCODE ----"
}

Run-Vox "perfOFF" "off"
Run-Vox "perfON"  "cuda_perf"
Write-Output "`n对比 run_perfOFF.log / run_perfON.log 的 RTF 与 wall time。"
