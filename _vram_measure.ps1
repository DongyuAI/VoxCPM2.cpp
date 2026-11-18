# 精确测量 VoxCPM2 Q8_0 合成期间峰值显存（后台高频采样 nvidia-smi）
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH
$text  = "大家好，欢迎收听今天的节目。人工智能技术正在快速发展，语音合成已经能够生成非常自然流畅的声音。今天我们一起讨论产品的未来方向，包括用户体验、性能优化、多语言支持以及自动化测试。感谢各位的参与，让我们共同努力，创造更好的明天。"
$out = "$root\TTS\VoxCPM2\vram_test.wav"
$log = "$root\TTS\VoxCPM2\vram_test.log"
$samples = "$root\TTS\VoxCPM2\vram_samples.csv"

# 后台采样器：每 50ms 记录一次显存
$job = Start-Job -ScriptBlock {
    param($path)
    $rows = @()
    for ($i=0; $i -lt 400; $i++) {
        $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) {
            $rows += "{0},{1}" -f (Get-Date -Format "HH:mm:ss.fff"), $v.Split("`n")[0].Trim()
        }
        Start-Sleep -Milliseconds 50
    }
    $rows | Set-Content $path
} -ArgumentList $samples

$env:GGML_CUDA_GRAPHS_PRE_AMPERE = "1"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe --task tts --family voxcpm2 --model $model --backend cuda `
    --text $text --out $out --threads 8 --seed 1234 `
    --num-inference-steps 6 --metrics --log-file $log `
    --session-option "perf_mode=cuda_perf" `
    --session-option "audiovae_tile_frames=1" `
    --session-option "dit_graph_arena_mb=256" `
    --session-option "lm_step_graph_arena_mb=256" `
    --session-option "audiovae_graph_arena_mb=256" `
    --session-option "audiovae_encoder_graph_arena_mb=256" 2>&1 |
    Select-String -Pattern "RTF|wall|audio duration|out of memory|failed"
$sw.Stop()
Remove-Item Env:\GGML_CUDA_GRAPHS_PRE_AMPERE -ErrorAction SilentlyContinue
Stop-Job $job -ErrorAction SilentlyContinue
Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
Remove-Job $job -Force -ErrorAction SilentlyContinue

$vals = Get-Content $samples -ErrorAction SilentlyContinue | ForEach-Object { [int]($_ -split ',')[1] }
if ($vals) {
    Write-Output "VRAM samples=$($vals.Count) min=$(($vals|Measure-Object -Minimum).Minimum)MiB max=$(($vals|Measure-Object -Maximum).Maximum)MiB"
}
Write-Output "WALL=$($sw.ElapsedMilliseconds)ms exit=$LASTEXITCODE"
