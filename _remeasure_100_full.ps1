# VoxCPM.cpp 完整 100 字重测（CLI，自动 natural_max_len，无 server 128 cap 截断）
# 修正此前 server 模式 service_cap=128 把 688 帧砍到 128 帧的错误
$ErrorActionPreference = "Continue"
$env:TEMP = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:TMP  = "F:\Tsinglin_Voice\.cache\audiocpp-temp"

$root = "F:\Tsinglin_Voice\TTS\VoxCPM.cpp"
Set-Location $root
$env:PATH = "$root\build-cuda3\bin;" + $env:PATH

$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

Write-Output "=== 完整 100 字 CLI 重测 x2 (q8_0 / 6步 / CUDA) ==="
for ($n=1; $n -le 2; $n++) {
    $out = "$root\wav100_cli_full_$n.wav"
    $log = "$root\wav100_cli_full_$n.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & "$root\build-cuda3\examples\voxcpm_tts.exe" `
        --model-path "$root\models\voxcpm-0.5b-q8_0-audiovae-f16.gguf" `
        --prompt-audio "$root\examples\tai_yi_xian_ren.wav" `
        --prompt-text "对，这就是我，万人敬仰的太乙真人。" `
        --text $text --output $out --backend cuda --threads 8 `
        --inference-timesteps 6 2>&1 | Tee-Object -FilePath $log | Out-Null
    $sw.Stop()
    $sz = (Get-Item $out -ErrorAction SilentlyContinue).Length
    $gen = Select-String -Path $log -Pattern "generated frames|generated_frames|decode loop done|Total:|AudioVAE decode|Model inference|Stop token" 2>$null
    Write-Output ("run{0}: wall={1}ms size={2}B" -f $n, $sw.ElapsedMilliseconds, $sz)
    $gen | ForEach-Object { Write-Output ("   " + $_.Line.Trim()) }
}
