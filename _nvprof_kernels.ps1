# 用 nvprof 抓 VoxCPM2 DiT 运行时真实 CUDA kernel，确认 matmul 是否走 DP4A (MMQ kernel)
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$exe = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$nvprof = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvprof.exe"
$env:PATH = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin;" + $env:PATH
$text = "你好，欢迎测试语音合成。"
$out = "F:\Tsinglin_Voice\TTS\VoxCPM2\prof_out.wav"
$log = "F:\Tsinglin_Voice\TTS\VoxCPM2\prof_nvprof.txt"

$args = @(
    "--task","tts","--family","voxcpm2","--model",$model,"--backend","cuda",
    "--text",$text,"--out",$out,"--threads","8","--seed","1234",
    "--num-inference-steps","6","--log-file","F:\Tsinglin_Voice\TTS\VoxCPM2\prof_run.log",
    "--session-option","perf_mode=cuda_perf",
    "--session-option","audiovae_tile_frames=1",
    "--session-option","dit_graph_arena_mb=256",
    "--session-option","lm_step_graph_arena_mb=256",
    "--session-option","audiovae_graph_arena_mb=256",
    "--session-option","audiovae_encoder_graph_arena_mb=256"
)
if (Test-Path $log) { Remove-Item $log -Force }
Write-Output "Running nvprof (kernel name + gpu time summary)..."
& $nvprof --concurrent-kernel-events on --log-file $log --profile-from-start on --metrics gpu__time_duration.sum $exe @args 2>&1 | Select-Object -Last 5
Write-Output "EXIT=$LASTEXITCODE"
if (Test-Path $log) {
    Write-Output "`n=== kernel summary (含 DP4A/MMQ 关键字) ==="
    Get-Content $log | Select-String -Pattern "mmq|mul_mat|dp4a|q8|cublas|gat|silu|soft_max|Name|GPU" | Select-Object -First 60
}
