# 自编译版参数精调：threads / mem_saver 对 DiT compute 的影响（profile 版）
# 全部锁 6 步 / seed1234 / perf_mode=cuda_perf / graph ON，纯性能参数，不动数值
$ErrorActionPreference = "Continue"
$env:TEMP = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:TMP  = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$exe = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe"
$model = "F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "F:\Tsinglin_Voice\TTS\audio.cpp\build\windows-cuda-release\bin;" + $env:PATH
$env:VOXCPM2_DIT_PROFILE = "1"
$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"
$text = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"

function Trial($name, $threads, [string[]]$extra) {
    $out = "F:\Tsinglin_Voice\TTS\VoxCPM2\th_$name.wav"
    $log = "F:\Tsinglin_Voice\TTS\VoxCPM2\th_$name.log"
    & $exe --task tts --family voxcpm2 --model $model --backend cuda `
      --text $text --out $out --threads $threads --seed 1234 `
      --num-inference-steps 6 --metrics --log-file $log `
      --session-option "perf_mode=cuda_perf" `
      --session-option "audiovae_tile_frames=1" `
      --session-option "dit_graph_arena_mb=256" `
      --session-option "lm_step_graph_arena_mb=256" `
      --session-option "audiovae_graph_arena_mb=256" `
      --session-option "audiovae_encoder_graph_arena_mb=256" @extra 2>&1 | Out-Null
    $all = Get-Content $log -Raw -ErrorAction SilentlyContinue
    $rtf = [regex]::Match($all, "metrics\.rtf=([0-9.]+)").Groups[1].Value
    $compute = ([regex]::Matches($all, "voxcpm2\.dit\.compute_ms ([0-9.]+)") | ForEach-Object { [double]$_.Groups[1].Value } | Measure-Object -Sum).Sum
    $sha = if (Test-Path $out) { (Get-FileHash $out -Algorithm SHA256).Hash.Substring(0,12) } else { "NONE" }
    Write-Output ("{0,-14} threads={1,-3} RTF={2,-9} DiT_compute={3,8:N0}ms sha={4}" -f $name,$threads,$rtf,$compute,$sha)
}

Trial "t1" 1 @()
Trial "t4" 4 @()
Trial "t8" 8 @()
Trial "t16" 16 @()
Trial "t8_nosaver" 8 @("--session-option", "mem_saver=false")
