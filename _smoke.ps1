# 快速冒烟测试：VoxCPM2 Q8_0 单文件GGUF 在 GTX1070 Pascal 上能否加载并合成
$ErrorActionPreference = "Continue"
$root  = "F:\Tsinglin_Voice"
$exe   = "$root\Bin\audiocpp\audiocpp_cli.exe"
$model = "$root\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
$env:PATH = "$root\Bin\audiocpp;" + $env:PATH

if (-not (Test-Path $model)) { Write-Output "FATAL: model not found: $model"; exit 1 }
$sz = (Get-Item $model).Length
Write-Output "MODEL SIZE: $sz bytes"

$log = "$root\TTS\VoxCPM2\smoke.log"
$out = "$root\TTS\VoxCPM2\smoke.wav"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe --task tts --family voxcpm2 --model $model --backend cuda `
    --text "你好，这是一次测试。" --out $out --threads 8 --seed 1234 `
    --num-inference-steps 6 --metrics --log-file $log 2>&1 |
    Select-String -Pattern "compute capability|Using backend|RTF|wall|audio duration|error|failed|VoxCPM2|voxcpm2|load|Cannot|unsupported"
$sw.Stop()
Write-Output "SMOKE wall=$($sw.ElapsedMilliseconds)ms exit=$LASTEXITCODE out=$((Get-Item $out -ErrorAction SilentlyContinue).Length)B"
Write-Output "FULL LOG: $log"
