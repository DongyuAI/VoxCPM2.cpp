# audio.cpp 基线构建脚本：sm_61 / CUDA 12.6 / 仅 voxcpm2 / 缓存全在 F 盘
# 用法: & "F:\Tsinglin_Voice\TTS\audio.cpp\_build_baseline.ps1"
$ErrorActionPreference = "Continue"

# 缓存严禁写 C 盘
$env:TEMP = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:TMP  = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:MODELSCOPE_CACHE = "F:\Tsinglin_Voice\.cache\modelscope"
$env:HF_HOME = "F:\Tsinglin_Voice\.cache\huggingface"
foreach ($d in @($env:TEMP,$env:TMP,$env:MODELSCOPE_CACHE,$env:HF_HOME)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$root = "F:\Tsinglin_Voice\TTS\audio.cpp"
Set-Location $root

$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

Write-Output "=== CONFIGURE (sm_61, only voxcpm2, cache->F) ==="
& "$root\scripts\build_windows.ps1" `
    -Preset "windows-cuda-release" `
    -Target "audiocpp_cli" `
    -CudaArchitectures "61-real" `
    -ModelSet "custom" `
    -Models "voxcpm2" `
    -NativeCpu "OFF" `
    -VsInstall $vs `
    -ConfigureOnly `
    -Jobs 3 2>&1 | Tee-Object -FilePath "$root\configure_baseline.log"

Write-Output "CONFIGURE_EXIT=$LASTEXITCODE"
