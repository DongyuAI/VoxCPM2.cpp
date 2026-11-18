# VoxCPM2 针对性参数扫描（纯性能参数，不碰步数/精度/guidance/CFG）
# 目标：找到 VoxCPM2 自己的最优配置，而非套用 VoxCPM 0.5 的应急配置
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"

function Trial($name, [string[]]$opts) {
    $out = "$root\TTS\VoxCPM2\scan_$name.wav"
    $log = "$root\TTS\VoxCPM2\scan_$name.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $m = & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --num-inference-steps 6 --metrics --log-file $log `
        --session-option "perf_mode=cuda_perf" @opts 2>&1
    $sw.Stop()
    $all = ($m | Out-String)
    $rtfM = [regex]::Match($all, "metrics\.rtf=([0-9.]+)")
    $wallM = [regex]::Match($all, "metrics\.wall_ms=([0-9.]+)")
    $rtf = if ($rtfM.Success) { $rtfM.Groups[1].Value } else { "?" }
    $wall = if ($wallM.Success) { $wallM.Groups[1].Value } else { "?" }
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash.Substring(0,16) } else { "NONE" }
    Write-Output ("{0,-22} wall={1,9}ms  RTF={2,-9} sha={3}" -f $name, $wall, $rtf, $sha)
}

# 基线：当前用的配置（0.5套来的）
$base = @(
  "--session-option","audiovae_tile_frames=1",
  "--session-option","dit_graph_arena_mb=256",
  "--session-option","lm_step_graph_arena_mb=256",
  "--session-option","audiovae_graph_arena_mb=256",
  "--session-option","audiovae_encoder_graph_arena_mb=256"
)
Trial "base_tile1_small" $base

# S1: tile=0（全量解码，官方默认）—— 之前 OOM 是因为显存被桌面占了1.27G，现在线程8可能仍OOM，测试确认
Trial "tile0" @(
  "--session-option","audiovae_tile_frames=0",
  "--session-option","dit_graph_arena_mb=256",
  "--session-option","lm_step_graph_arena_mb=256",
  "--session-option","audiovae_graph_arena_mb=256",
  "--session-option","audiovae_encoder_graph_arena_mb=256"
)

# S2: mem_saver=false（保留图资源，避免每步释放重建）
Trial "memsaver_off" ($base + @("--session-option","mem_saver=false"))

# S3: 默认 graph arena（1024），只 tile=1
Trial "tile1_fullarena" @("--session-option","audiovae_tile_frames=1")

# S4: audiovae_latent_capacity 预分配（避免按请求建图）
Trial "tile1_latentcap" ($base + @("--session-option","audiovae_latent_capacity=600"))

# S5: tile=4 折中
Trial "tile4" @(
  "--session-option","audiovae_tile_frames=4",
  "--session-option","dit_graph_arena_mb=256",
  "--session-option","lm_step_graph_arena_mb=256",
  "--session-option","audiovae_graph_arena_mb=256",
  "--session-option","audiovae_encoder_graph_arena_mb=256"
)

Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
