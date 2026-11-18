# 100字合成显存峰值 + 端到端耗时实测（q8_0 / 6步 / server缓存音色）
$ErrorActionPreference = "Continue"
$root = "F:\Tsinglin_Voice\TTS\VoxCPM.cpp"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$body = @{ model="voxcpm"; input=$text; voice="taiyi"; response_format="wav" } | ConvertTo-Json

# 后台采样显存（每 50ms）
$sampler = Start-Job -ScriptBlock {
    $vals = @()
    $end = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $end) {
        $m = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($m) { $vals += [int]($m.Trim().Split("`n")[0]) }
        Start-Sleep -Milliseconds 50
    }
    return $vals
}

Write-Output "显存基线(加载后空闲): $((& nvidia-smi --query-gpu=memory.used --format=csv,noheader).Trim()) MiB"
Write-Output "`n=== 100字合成 x3 ==="
for ($n=1; $n -le 3; $n++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $code = $null
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/audio/speech" `
            -Method Post -Body $body -ContentType "application/json" `
            -OutFile "$root\wav100_$n.wav" -TimeoutSec 120 -UseBasicParsing
        $code = $resp.StatusCode
    } catch { $code = "ERR: $($_.Exception.Message)" }
    $sw.Stop()
    $sz = (Get-Item "$root\wav100_$n.wav" -ErrorAction SilentlyContinue).Length
    $mid = & nvidia-smi --query-gpu=memory.used --format=csv,noheader
    Write-Output ("run{0}: HTTP={1} wall={2}ms size={3}B vram_now={4} MiB" -f $n, $code, $sw.ElapsedMilliseconds, $sz, $mid.Trim())
    Start-Sleep -Milliseconds 300
}

$vals = Receive-Job $sampler -Wait
Remove-Job $sampler
if ($vals.Count -gt 0) {
    Write-Output "`n=== 显存采样（合成期间，样本数 $($vals.Count)）==="
    Write-Output ("min={0} MiB  max={1} MiB  avg={2} MiB" -f ($vals | Measure-Object -Minimum).Minimum, ($vals | Measure-Object -Maximum).Maximum, [int](($vals | Measure-Object -Average).Average))
}
