# VoxCPM2 低显存诊断：尝试 tile 解码 + 缩减 graph workspace
# 硬约束：steps=6、seed=1234、不降精度（weight_type 保持 GGUF 原生 q8_0）
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH
$text = "你好，这是一次测试。"

function Trial($name, [string[]]$sessionOpts) {
    Write-Output "`n########## $name ##########"
    Write-Output ("OPTS: " + ($sessionOpts -join " "))
    $log = "$root\TTS\VoxCPM2\diag_$name.log"
    $out = "$root\TTS\VoxCPM2\diag_$name.wav"
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --num-inference-steps 6 --metrics --log-file $log `
        @sessionOpts 2>&1 |
        Select-String -Pattern "RTF|wall|audio duration|cudaMalloc|out of memory|failed|VoxCPM2 AudioVAE|error|loaded CUDA"
    Write-Output "---- $name exit=$LASTEXITCODE out=$((Get-Item $out -ErrorAction SilentlyContinue).Length)B ----"
}

# T1: tile 解码，每 tile 8 帧
Trial "tile8" @("--session-option", "audiovae_tile_frames=8")
# T2: tile 1 帧 + 缩减 host graph arena
Trial "tile1_small" @(
    "--session-option", "audiovae_tile_frames=1",
    "--session-option", "dit_graph_arena_mb=256",
    "--session-option", "lm_step_graph_arena_mb=256",
    "--session-option", "audiovae_graph_arena_mb=256",
    "--session-option", "audiovae_encoder_graph_arena_mb=256"
)
