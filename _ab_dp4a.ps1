# Pascal DP4A 融合实验 A/B：对比 基线 vs 融合版
# 锁 6 步 / seed1234 / Q8_0 / cuda_perf，验证 SHA 数值等价 + RTF/DiT compute 提速
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$exeFused = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe"   # 改了 Pascal gate
$exeBase  = "F:\Tsinglin_Voice\Bin\audiocpp\audiocpp_cli.exe"                                    # 0.9.5 预编译基线
$env:PATH = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin;" + $env:PATH
$env:VOXCPM2_DIT_PROFILE = "1"
$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"

function Trial($name, $exe) {
    $out = "F:\Tsinglin_Voice\TTS\VoxCPM2\dp4a_$name.wav"
    $log = "F:\Tsinglin_Voice\TTS\VoxCPM2\dp4a_$name.log"
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
    Write-Output ("{0,-10} RTF={1,-9} DiT_compute={2,8:N0}ms SHA={3}" -f $name,$rtf,$compute,$sha.Substring(0,[Math]::Min(16,$sha.Length)))
    return $sha
}

Write-Output "=== Pascal DP4A 融合 A/B (6步/seed1234/Q8_0) ==="
$s1 = Trial "base"  $exeBase
$s2 = Trial "fused" $exeFused
Write-Output "`nSHA equal: $($s1 -eq $s2)"
