# 验证 num_inference_steps 参数真实生效：1步 vs 6步（仅诊断，不用于质量基准）
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\_env_cache.ps1"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH
$text = "你好，这是一次测试。"

foreach ($n in 1, 6) {
    Write-Output "`n########## steps=$n ##########"
    $log = "$root\TTS\VoxCPM2\steps_$n.log"
    $out = "$root\TTS\VoxCPM2\steps_$n.wav"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
        --text $text --out $out --threads 8 --seed 1234 `
        --request-option "num_inference_steps=$n" --metrics --log-file $log `
        --session-option "audiovae_tile_frames=1" `
        --session-option "dit_graph_arena_mb=256" `
        --session-option "lm_step_graph_arena_mb=256" `
        --session-option "audiovae_graph_arena_mb=256" `
        --session-option "audiovae_encoder_graph_arena_mb=256" 2>&1 |
        Select-String -Pattern "RTF|wall|audio duration|error|failed"
    $sw.Stop()
    Write-Output "---- steps=$n wall=$($sw.ElapsedMilliseconds)ms exit=$LASTEXITCODE ----"
}
