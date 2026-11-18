# device 常驻优化 A/B：base(每步上传cond/mu) vs devres(仅变化时上传)
# 数值必须完全等价（同样的float payload到同样的tensor）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$bdir = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin"
$env:PATH = "$bdir;" + $env:PATH
$env:VOXCPM2_DIT_PROFILE = "1"
$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"

$script:s1=""; $script:s2=""; $script:u1=0; $script:u2=0; $script:c1=0; $script:c2=0; $script:w1=0; $script:w2=0
function Trial($name, $exe) {
    $out = "F:\Tsinglin_Voice\TTS\VoxCPM2\devres_$name.wav"
    $log = "F:\Tsinglin_Voice\TTS\VoxCPM2\devres_$name.log"
    if (Test-Path $out) { Remove-Item $out -Force }
    if (Test-Path $log) { Remove-Item $log -Force }
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
      --text $text --out $out --threads 8 --seed 1234 `
      --num-inference-steps 6 --metrics --log-file $log `
      --session-option "perf_mode=cuda_perf" `
      --session-option "audiovae_tile_frames=1" `
      --session-option "dit_graph_arena_mb=256" `
      --session-option "lm_step_graph_arena_mb=256" `
      --session-option "audiovae_graph_arena_mb=256" `
      --session-option "audiovae_encoder_graph_arena_mb=256" 2>&1 | Out-Null
    $all = Get-Content $log -Raw -ErrorAction SilentlyContinue
    $up = ([regex]::Matches($all,"voxcpm2\.dit\.upload_ms ([0-9.]+)")|%{[double]$_.Groups[1].Value}|Measure-Object -Sum).Sum
    $cp = ([regex]::Matches($all,"voxcpm2\.dit\.compute_ms ([0-9.]+)")|%{[double]$_.Groups[1].Value}|Measure-Object -Sum).Sum
    $wl = [double]([regex]::Match($all,"session\.wall_ms ([0-9.]+)").Groups[1].Value)
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash } else { "NONE" }
    if ($name -eq "base") {$script:s1=$sha;$script:u1=$up;$script:c1=$cp;$script:w1=$wl} else {$script:s2=$sha;$script:u2=$up;$script:c2=$cp;$script:w2=$wl}
    Write-Output ("{0,-6} upload={1,6:N1}ms compute={2,6:N0}ms wall={3,7:N0}ms SHA={4}" -f $name,$up,$cp,$wl,$sha.Substring(0,16))
}
Write-Output "=== device 常驻 A/B (6步/seed1234/Q8_0/cuda_perf/graph ON) ==="
Trial "base"  "$bdir\audiocpp_base.exe"
Trial "devres" "$bdir\audiocpp_devres.exe"
Write-Output "`nSHA_EQUAL=$($script:s1 -eq $script:s2)"
Write-Output ("upload: {0:N1} -> {1:N1}ms (saved {2:N1}ms)" -f $script:u1,$script:u2,($script:u1-$script:u2))
Write-Output ("wall:   {0:N0} -> {1:N0}ms (saved {2:N0}ms, {3:N1}%)" -f $script:w1,$script:w2,($script:w1-$script:w2),(100*($script:w1-$script:w2)/$script:w1))
