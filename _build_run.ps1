# 编译 audio.cpp（sm_61 / voxcpm2），缓存全在 F 盘
$ErrorActionPreference = "Continue"
$env:TEMP = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:TMP  = "F:\Tsinglin_Voice\.cache\audiocpp-temp"
$env:MODELSCOPE_CACHE = "F:\Tsinglin_Voice\.cache\modelscope"
$env:HF_HOME = "F:\Tsinglin_Voice\.cache\huggingface"

$root = "F:\Tsinglin_Voice\TTS\audio.cpp"
Set-Location $root
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

$vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkver = "10.0.22621.0"
$cmakeBin = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$ninjaBin = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

$env:PATH = "$vc\bin\Hostx64\x64;$sdk\bin\$sdkver\x64;$ninjaBin;$cmakeBin;$env:CUDA_PATH\bin;" + $env:PATH
$env:INCLUDE = "$vc\include;$sdk\Include\$sdkver\um;$sdk\Include\$sdkver\ucrt;$sdk\Include\$sdkver\shared;$sdk\Include\$sdkver\winrt;$env:CUDA_PATH\include"
$env:LIB = "$vc\lib\x64;$sdk\Lib\$sdkver\um\x64;$sdk\Lib\$sdkver\ucrt\x64;$env:CUDA_PATH\lib\x64"

$BUILD = "build\windows-cuda-release"
Write-Output "=== BUILD audiocpp_cli -j6 (用户指定) ==="
& "$cmakeBin\cmake.exe" --build $BUILD --target audiocpp_cli -j 6 2>&1 | Tee-Object -FilePath "$root\build_dp4a.log"
Write-Output "BUILD_EXIT=$LASTEXITCODE"
