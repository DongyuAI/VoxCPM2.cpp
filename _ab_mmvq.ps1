# 严谨 A/B：同源码，唯一差异 = Pascal MMVQ 融合 gate
# base  = 上游原样(cc<=PASCAL 禁用 fused MMVQ)
# fused = 去掉 gate(Pascal 启用 DP4A fused MMVQ)
# 锁 6步/seed1234/Q8_0/cuda_perf/graph ON
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$bdir = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin"
$env:PATH = "$bdir;" + $env:PATH
$env:VOXCPM2_DIT_PROFILE = "1"
$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"

function Trial($name, $exe) {
    $out = "F:\Tsinglin_Voice\TTS\VoxCPM2\mmvq_$name.wav"
    $log = "F:\Tsinglin_Voice\TTS\VoxCPM2\mmvq_$name.log"
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
    $rtf = [regex]::Match($all, "metrics\.rtf=([0-9.]+)").Groups[1].Value
    $compute = ([regex]::Matches($all, "voxcpm2\.dit\.compute_ms ([0-9.]+)") | ForEach-Object { [double]$_.Groups[1].Value } | Measure-Object -Sum).Sum
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash } else { "NONE" }
    [pscustomobject]@{ name=$name; rtf=$rtf; compute=$compute; sha=$sha }
    Write-Output ("{0,-8} RTF={1,-9} DiT_compute={2,8:N0}ms SHA={3}" -f $name,$rtf,$compute,$sha.Substring(0,16))
}

Write-Output "=== Pascal MMVQ 融合 A/B (6步/seed1234/Q8_0/cuda_perf/graph ON) ==="
$a = Trial "base"  "$bdir\audiocpp_base.exe"
$b = Trial "fused" "$bdir\audiocpp_fused.exe"
Write-Output "`nSHA_EQUAL=$($a.sha -eq $b.sha)"
if ($a.compute -gt 0 -and $b.compute -gt 0) {
  Write-Output ("DiT_compute base={0:N0}ms fused={1:N0}ms delta={2:N1}%" -f $a.compute,$b.compute,(100*($b.compute-$a.compute)/$a.compute))
}
