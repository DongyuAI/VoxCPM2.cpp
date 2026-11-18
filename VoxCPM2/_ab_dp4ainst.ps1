# DP4A 硬件指令收益实测：base(硬件__dp4a) vs scalar(强制4次标量乘加)
# 唯一差异 = common.cuh 里 ggml_cuda_dp4a 是否用 __dp4a 指令
# 数值必须完全等价（同样乘加，只是不用硬件指令）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$bdir = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin"
$env:PATH = "$bdir;" + $env:PATH
$env:VOXCPM2_DIT_PROFILE = "1"
$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"

$script:s1 = ""; $script:s2 = ""; $script:c1 = 0; $script:c2 = 0

function Trial($name, $exe) {
    $out = "F:\Tsinglin_Voice\TTS\VoxCPM2\dp4ainst_$name.wav"
    $log = "F:\Tsinglin_Voice\TTS\VoxCPM2\dp4ainst_$name.log"
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
    $compute = ([regex]::Matches($all, "voxcpm2\.dit\.compute_ms ([0-9.]+)") | ForEach-Object { [double]$_.Groups[1].Value } | Measure-Object -Sum).Sum
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash } else { "NONE" }
    if ($name -eq "hw") { $script:s1=$sha; $script:c1=$compute } else { $script:s2=$sha; $script:c2=$compute }
    Write-Output ("{0,-8} DiT_compute={1,8:N0}ms SHA={2}" -f $name,$compute,$sha.Substring(0,16))
}

Write-Output "=== DP4A 指令收益 A/B (6步/seed1234/Q8_0/cuda_perf/graph ON) ==="
Trial "hw"     "$bdir\audiocpp_base.exe"
Trial "scalar" "$bdir\audiocpp_scalar.exe"
Write-Output "`nSHA_EQUAL=$($script:s1 -eq $script:s2)"
Write-Output ("DiT_compute hw={0:N0}ms scalar={1:N0}ms DP4A_speedup={2:N1}%" -f $script:c1,$script:c2,(100*($script:c2-$script:c1)/$script:c1))
