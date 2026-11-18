"""稳健解析 GGUF v3：dump 所有 tensor 的 name/shape/dtype/offset，按前缀分组统计字节数。
缓存全在 F 盘，不写 C 盘。
"""
import os, struct, collections

os.environ["TMP"] = r"F:\Tsinglin_Voice\.cache\audiocpp-temp"
os.environ["TEMP"] = r"F:\Tsinglin_Voice\.cache\audiocpp-temp"

PATH = r"F:\Tsinglin_Voice\TTS\VoxCPM2\models_ms\VoxCPM2-GGUF\voxcpm2-q8_0.gguf"
DTYPE_SIZE = {0:0.5,1:4,2:8,3:2,4:4,5:8,6:4,7:1,8:1,9:2,10:2,
              11:2,12:4,13:4,14:8,16:0.5625,17:0.5625,18:0.5625,19:0.5625,
              20:0.5625,21:0.5625,22:0.5625,23:0.5625,24:0.5,25:0.5,26:0.5,
              27:0.5,28:8,29:0.5,30:1,32:1,33:1,34:1,35:1,36:1}
DTYPE_NAME = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
              9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",
              28:"F64",30:"BF16"}

f = open(PATH, "rb")
magic = f.read(4)
ver = struct.unpack("<I", f.read(4))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
print(f"magic={magic} ver={ver} n_kv={n_kv} n_tensors={n_tensors}")

def read_str():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

# GGUF v3 value type codes (correct spec)
def skip_value(t):
    fixed = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    if t in fixed:
        f.read(fixed[t])
    elif t == 8:
        read_str()       # STRING
    elif t == 9:         # ARRAY: u64 n + u32 elem_type + data
        n = struct.unpack("<Q", f.read(8))[0]
        et = struct.unpack("<I", f.read(4))[0]
        if et == 8:
            for _ in range(n): read_str()
        elif et in fixed:
            f.read(n * fixed[et])
        else:
            raise ValueError(f"unexpected array elem type {et}")

for _ in range(n_kv):
    k = read_str()
    t = struct.unpack("<I", f.read(4))[0]
    skip_value(t)

# Now tensor info table (GGUF v3)
groups = collections.defaultdict(lambda: [0,0,collections.Counter()])
tensors = []
for i in range(n_tensors):
    name = read_str()
    ndim = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
    dtype = struct.unpack("<I", f.read(4))[0]
    offset = struct.unpack("<Q", f.read(8))[0]
    nelem = 1
    for d in dims: nelem *= d
    bpe = DTYPE_SIZE.get(dtype, 1)
    nbytes = int(nelem * bpe)
    tensors.append((name, dims, dtype, nbytes))
    # classify
    if name.startswith("base_lm"): g = "base_lm"
    elif name.startswith("residual_lm"): g = "residual_lm"
    elif name.startswith("feat_encoder"): g = "feat_encoder"
    elif name.startswith("feat_decoder.estimator.decoder"): g = "DiT_decoder"
    elif name.startswith("feat_decoder.estimator"): g = "DiT_head(time/cond/proj)"
    elif name.startswith("fsq") or name.startswith("enc_to_lm") or name.startswith("lm_to_dit") or name.startswith("res_to_dit") or name.startswith("fusion") or name.startswith("stop"): g = "projections"
    elif "audiovae" in name or "vae" in name.lower(): g = "AudioVAE"
    else: g = "other/"+name.split(".")[0]
    groups[g][0] += 1
    groups[g][1] += nbytes
    groups[g][2][DTYPE_NAME.get(dtype,str(dtype))] += nbytes

print("\n=== 按组统计 (tensor数, MB, dtype分布) ===")
tot = 0
for g in sorted(groups, key=lambda x:-groups[x][1]):
    cnt, nb, dt = groups[g]
    tot += nb
    dts = ", ".join(f"{k}:{v/1048576:.1f}MB" for k,v in dt.most_common(4))
    print(f"{g:28s} n={cnt:4d}  {nb/1048576:8.1f} MB  [{dts}]")
print(f"{'TOTAL':28s} {'':8s}  {tot/1048576:8.1f} MB")

# DiT decoder detail
print("\n=== DiT decoder 各层 tensor ===")
for name, dims, dtype, nb in tensors:
    if name.startswith("feat_decoder.estimator.decoder") and ("layers.0." in name or "layers." not in name):
        print(f"  {name:70s} {str(dims):20s} {DTYPE_NAME.get(dtype,dtype):6s} {nb/1048576:.2f}MB")
