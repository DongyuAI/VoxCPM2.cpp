# 统一缓存路径注入：严禁任何缓存写入 C 盘
# 用法：在所有会调用 audiocpp_cli / modelscope / huggingface 的脚本开头：
#        . "$PSScriptRoot\_env_cache.ps1"
#
# 背景：
#  - audio.cpp 的 materialize_gguf_sidecars() 用 std::filesystem::temp_directory_path()
#    / "audiocpp-gguf" / <fingerprint> 解压 tokenizer.json 等，Windows 下该标准函数读 TMP/TEMP。
#  - ModelScope snapshot_download 默认缓存到 ~/.cache/modelscope（C 盘）。
#  - HuggingFace 默认缓存到 ~/.cache/huggingface（C 盘）。
# 全部重定向到 F:\Tsinglin_Voice\.cache\ 下。

$CacheRoot = "F:\Tsinglin_Voice\.cache"

$env:TEMP = "$CacheRoot\audiocpp-temp"
$env:TMP  = "$CacheRoot\audiocpp-temp"
$env:MODELSCOPE_CACHE = "$CacheRoot\modelscope"
$env:HF_HOME = "$CacheRoot\huggingface"
$env:XDG_CACHE_HOME = "$CacheRoot\xdg"

foreach ($d in @($env:TEMP, $env:MODELSCOPE_CACHE, $env:HF_HOME, $env:XDG_CACHE_HOME)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
