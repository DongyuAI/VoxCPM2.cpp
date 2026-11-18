$vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkver = "10.0.22621.0"
$cmakeBin = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$ninjaBin = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

$env:PATH = "$vc\bin\Hostx64\x64;$sdk\bin\$sdkver\x64;$ninjaBin;$cmakeBin;" + $env:PATH
$env:INCLUDE = "$vc\include;$sdk\Include\$sdkver\um;$sdk\Include\$sdkver\ucrt;$sdk\Include\$sdkver\shared;$sdk\Include\$sdkver\winrt"
$env:LIB = "$vc\lib\x64;$sdk\Lib\$sdkver\um\x64;$sdk\Lib\$sdkver\ucrt\x64"
$env:VSCMD_VER = "17.11.0"
$env:VCToolsInstallDir = "$vc\"
$env:UniversalCRTSdkDir = "$sdk\"
$env:UCRTVersion = $sdkver
$env:WindowsSdkBinPath = "$sdk\bin\"
$env:WindowsSdkDir = "$sdk\"

Set-Location "F:\Tsinglin_Voice\TTS\VoxCPM.cpp"
$BUILD = "build-cuda3"
Write-Output "=== reconfigure CUDA sm_61 ($BUILD, reuse) ==="
& "$cmakeBin\cmake.exe" -B $BUILD -G Ninja `
  -DVOXCPM_CUDA=ON `
  -DVOXCPM_BUILD_BENCHMARK=OFF `
  -DVOXCPM_BUILD_TESTS=OFF `
  -DVOXCPM_NATIVE=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=61 `
  "-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG" `
  -DCMAKE_BUILD_TYPE=Release 2>&1 | Tee-Object -FilePath "F:\Tsinglin_Voice\TTS\VoxCPM.cpp\configure.log"
Write-Output "CONFIGURE_EXIT=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 2 }

Write-Output "=== build -j1 (reuse compiled objects) ==="
& "$cmakeBin\cmake.exe" --build $BUILD -j 1 2>&1 | Tee-Object -FilePath "F:\Tsinglin_Voice\TTS\VoxCPM.cpp\build2.log"
Write-Output "BUILD_EXIT=$LASTEXITCODE"
