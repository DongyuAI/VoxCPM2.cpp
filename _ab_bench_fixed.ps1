# 严格等步数 A/B：固定 40 步 + ignore stop，消除随机 stop 时机干扰
# 质量参数完全不动：q8_0 / 6 步 / 同文本。仅测每步耗时（it/s）
$ErrorActionPreference = "Continue"
$root = "F:\Tsinglin_Voice\TTS\VoxCPM.cpp"
$model   = "$root\models\voxcpm-0.5b-q8_0-audiovae-f16.gguf"
$promptA = "$root\examples\tai_yi_xian_ren.wav"
$promptT = "对，这就是我，万人敬仰的太乙真人。"
$text    = "大家好，我现在正在大可奇奇体验AI科技。"

$targets = @(
    @{ name="graphOFF"; exe="$root\build-cuda3\examples\voxcpm_tts.exe";     bin="$root\build-cuda3\bin" },
    @{ name="graphON";  exe="$root\build-cudagraph\examples\voxcpm_tts.exe"; bin="$root\build-cudagraph\bin" }
)

foreach ($t in $targets) {
    $env:PATH = "$($t.bin);" + $env:PATH
    $env:VOXCPM_BENCHMARK_DECODE_STEPS = "40"
    $env:VOXCPM_BENCHMARK_IGNORE_STOP = "1"
    $env:VOXCPM_LOG_DECODE_TIMING = "1"
    Write-Output "`n########## $($t.name) : 固定40步 + ignore_stop ##########"
    $out = "$root\bench_$($t.name).wav"
    $log = "$root\bench_$($t.name).log"
    & $t.exe --model-path $model --prompt-audio $promptA --prompt-text $promptT `
        --text $text --output $out --backend cuda --inference-timesteps 6 --threads 8 2>&1 |
        Tee-Object -FilePath $log |
        Select-String -Pattern "benchmark_steps|CUDA graph warmup|Model inference|Total:|RTF|decode\] 40/|decode\] 39/"
    Write-Output "---- $($t.name) exit=$LASTEXITCODE ----"
}
Write-Output "`n对比 bench_graphOFF.log / bench_graphON.log 的 40 步 it/s 与 Model inference 时间。"
