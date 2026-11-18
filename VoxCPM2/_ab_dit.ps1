# 基线 vs DiT 优化版 无损 A/B
# 锁 6 步 / seed1234 / 100字 / Q8_0 / 同一低显存配置
# 对比：wall / RTF / generator_ms / 输出 SHA256
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\..\TTS\VoxCPM2\_env_cache.ps1"
$root  = "F:\Tsinglin_Voice"
$baseExe = "$root\Bin\audiocpp\audiocpp_cli.exe"
$optExe  = "$root\TTS\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$common = @(
  "--session-option","audiovae_tile_frames=1",
  "--session-option","dit_graph_arena_mb=256",
  "--session-option","lm_step_graph_arena_mb=256",
  "--session-option","audiovae_graph_arena_mb=256",
  "--session-option","audiovae_encoder_graph_arena_mb=256",
  "--session-option","perf_mode=cuda_perf"
)

function Run($label, $exe) {
    if (-not (Test-Path $exe)) { Write-Output "SKIP $label: no exe $exe"; return }
    Write-Output "`n########## $label ##########"
    Write-Output "EXE: $exe"
    $out = "$root\TTS\VoxCPM2\opt_$label.wav"
    $log = "$root\TTS\VoxCPM2\opt_$label.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
      --text $text --out $out --threads 8 --seed 1234 `
      --num-inference-steps 6 --metrics --log-file $log @common 2>&1 |
      Select-String -Pattern "RTF|wall|audio|error|failed|voxcpm2.dit"
    $sw.Stop()
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash } else { "NONE" }
    $gen = (Select-String -Path $log -Pattern "generator_ms ([\d.]+)" -ErrorAction SilentlyContinue).Matches.Groups[1].Value
    Write-Output ("---- {0}: wall={1}ms generator={2}ms sha={3} ----" -f $label,$sw.ElapsedMilliseconds,$gen,$sha.Substring(0,[Math]::Min(24,$sha.Length)))
}

Run "BASE" $baseExe
Run "OPT"  $optExe
