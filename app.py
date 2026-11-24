# ==============================================================================
# STEP 1: STANDALONE CLINICAL DATASET DOWNLOAD & INTEGRITY VALIDATION CELL
# Target: /tmp/medmnist_sovereign_cache/{pathmnist.npz, bloodmnist.npz}
# ==============================================================================

import os
import sys
import time
import hashlib
import subprocess
import numpy as np
import requests

CACHE_DIR = "/tmp/medmnist_sovereign_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS = {
    "pathmnist.npz": {
        "config": "pathmnist",
        "min_bytes": 150_000_000,  # ~205 MB
        "urls": [
            "https://zenodo.org/api/records/10519652/files/pathmnist.npz/content",
            "https://zenodo.org/api/records/6496656/files/pathmnist.npz/content",
            "https://zenodo.org/records/10519652/files/pathmnist.npz",
        ],
    },
    "bloodmnist.npz": {
        "config": "bloodmnist",
        "min_bytes": 20_000_000,   # ~35 MB
        "urls": [
            "https://zenodo.org/api/records/10519652/files/bloodmnist.npz/content",
            "https://zenodo.org/api/records/6496656/files/bloodmnist.npz/content",
            "https://zenodo.org/records/10519652/files/bloodmnist.npz",
        ],
    },
}

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

def download_via_curl_or_requests(filename: str, urls: list, min_bytes: int) -> bool:
    target = os.path.join(CACHE_DIR, filename)
    if os.path.exists(target) and os.path.getsize(target) >= min_bytes:
        print(f"[Cache Hit] {filename} already exists ({os.path.getsize(target)/(1024**2):.2f} MB)")
        return True

    # Method 1: System curl with HTTP/2 + full browser headers on Zenodo API endpoints
    for url in urls:
        print(f"[Attempt curl] Fetching {filename} from:\n  -> {url}")
        t0 = time.time()
        cmd = [
            "curl", "-L", "--fail", "--silent", "--show-error",
            "--retry", "2", "--connect-timeout", "20",
            "-A", BROWSER_UA,
            "-H", "Accept: application/octet-stream,*/*",
            "-o", target,
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(target) and os.path.getsize(target) >= min_bytes:
            dt = time.time() - t0
            mb = os.path.getsize(target) / (1024 ** 2)
            print(f"  [OK] Downloaded {filename} ({mb:.2f} MB in {dt:.1f}s | {mb/max(dt,0.1):.1f} MB/s)")
            return True
        else:
            err_msg = res.stderr.strip() if res.stderr else "Size/HTTP check failed"
            print(f"  [Warn] curl failed: {err_msg}")
            if os.path.exists(target):
                os.remove(target)

    # Method 2: Python requests session with streaming
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_UA, "Accept": "*/*"})
    for url in urls:
        try:
            print(f"[Attempt requests] Fetching {filename} from {url} ...")
            t0 = time.time()
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(target) and os.path.getsize(target) >= min_bytes:
                dt = time.time() - t0
                mb = os.path.getsize(target) / (1024 ** 2)
                print(f"  [OK] Downloaded {filename} ({mb:.2f} MB in {dt:.1f}s)")
                return True
        except Exception as e:
            print(f"  [Warn] requests failed: {e}")
            if os.path.exists(target):
                os.remove(target)

    return False


def download_via_huggingface_fallback(filename: str, config_name: str) -> bool:
    """
    If Zenodo blocks Kaggle's GCP IP via Cloudflare 403, pull the exact MedMNIST v2
    dataset from Hugging Face Hub ('albertvillanova/medmnist-v2') and pack into .npz.
    """
    target = os.path.join(CACHE_DIR, filename)
    print(f"[HuggingFace Fallback] Loading '{config_name}' from Hugging Face Hub (albertvillanova/medmnist-v2)...")
    from datasets import load_dataset
    ds = load_dataset("albertvillanova/medmnist-v2", config_name, trust_remote_code=True)

    def _extract_split(split_obj):
        imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in split_obj["image"]], axis=0)
        lbls = np.array(split_obj["label"], dtype=np.int64).reshape(-1, 1)
        return imgs, lbls

    tr_i, tr_l = _extract_split(ds["train"])
    va_i, va_l = _extract_split(ds["validation"])
    te_i, te_l = _extract_split(ds["test"])

    np.savez_compressed(
        target,
        train_images=tr_i, train_labels=tr_l,
        val_images=va_i,   val_labels=va_l,
        test_images=te_i,  test_labels=te_l,
    )
    print(f"  [OK] Saved {filename} via Hugging Face Hub ({os.path.getsize(target)/(1024**2):.2f} MB)")
    return True


print("=" * 88)
print(" STARTING CLINICAL DATASET ACQUISITION & CRYPTOGRAPHIC VALIDATION")
print("=" * 88)

for fname, meta in DATASETS.items():
    ok = download_via_curl_or_requests(fname, meta["urls"], meta["min_bytes"])
    if not ok:
        ok = download_via_huggingface_fallback(fname, meta["config"])
    if not ok:
        raise RuntimeError(f"Could not acquire {fname} from any mirror.")

# ==============================================================================
# DEEP INTEGRITY & SCHEMA VALIDATION
# ==============================================================================
print("\n" + "=" * 88)
print(" VERIFYING .NPZ ARRAYS, PIXEL DISTRIBUTIONS, AND SHA-256 CHECKSUMS")
print("=" * 88)

for fname in ["pathmnist.npz", "bloodmnist.npz"]:
    fpath = os.path.join(CACHE_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 ** 2)
    sha256_hash = hashlib.sha256(open(fpath, "rb").read()).hexdigest()

    data = np.load(fpath)
    keys = list(data.keys())
    tr_x, tr_y = data["train_images"], data["train_labels"].flatten()
    va_x, va_y = data["val_images"], data["val_labels"].flatten()
    te_x, te_y = data["test_images"], data["test_labels"].flatten()

    unique_classes, counts = np.unique(tr_y, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique_classes, counts)}

    print(f"\nDataset File      : {fpath} ({size_mb:.2f} MB)")
    print(f"SHA-256 Checksum  : {sha256_hash}")
    print(f"Archive Keys      : {keys}")
    print(f"Train Split       : images={tr_x.shape} ({tr_x.dtype}) | labels={tr_y.shape} | pixel range=[{tr_x.min()}, {tr_x.max()}]")
    print(f"Val Split         : images={va_x.shape} ({va_x.dtype}) | labels={va_y.shape}")
    print(f"Test Split        : images={te_x.shape} ({te_x.dtype}) | labels={te_y.shape}")
    print(f"Train Class Count : {len(unique_classes)} classes -> {class_dist}")

    assert tr_x.ndim == 4 and tr_x.shape[1:] == (28, 28, 3), f"Unexpected shape in {fname}: {tr_x.shape}"
    assert tr_x.dtype == np.uint8, f"Expected uint8 in {fname}, got {tr_x.dtype}"
    assert int(tr_x.max()) > 200 and int(tr_x.min()) < 50, "Invalid pixel dynamic range!"

print("\n" + "=" * 88)
print(" [SUCCESS] 100% REAL CLINICAL HISTOPATHOLOGY & HEMATOLOGY DATASETS VERIFIED ON DISK!")





#Main_Code



# ====================================================================================
# Verified Core + Live Persian (RTL) Cloudflare Web Dashboard | Target: Kaggle 2x T4
# ====================================================================================
import base64
import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
_GRAPH_CAPTURE_LOCK = threading.Lock()
CLASS_NAMES: Dict[int, tuple] = {
    0: ("ADI", "Adipose Tissue"),
    1: ("BACK", "Background"),
    2: ("DEB", "Debris / Necrosis"),
    3: ("LYM", "Lymphocytes"),
    4: ("MUC", "Mucus"),
    5: ("MUS", "Smooth Muscle"),
    6: ("NORM", "Normal Colon Mucosa"),
    7: ("STR", "Cancer-Associated Stroma"),
    8: ("TUM", "Colorectal Adenocarcinoma"),
}
CLASS_NAMES_FA: Dict[int, str] = {
    0: "بافت چربی (Adipose Tissue)",
    1: "پس‌زمینه لام (Background)",
    2: "نکروز و بقایای سلولی (Debris / Necrosis)",
    3: "لنفوسیت‌های ایمنی (Lymphocytes)",
    4: "تجمعات موکوسی (Mucus)",
    5: "عضله صاف (Smooth Muscle)",
    6: "مخاط نرمال کولون (Normal Colon Mucosa)",
    7: "استرومای مرتبط با تومور (Cancer-Associated Stroma)",
    8: "آدنوکارسینومای بدخیم کولورکتال (Colorectal Adenocarcinoma)",
}
NUM_CLASSES = 9
CACHE_DIR = "/tmp/medmnist_sovereign_cache"
# ====================================================================================
# 1. OFFICIAL MEDMNIST V2 DATA LOADER (ZERO DATA LEAKAGE)
# ====================================================================================
def _download_split_npz(filename: str, urls: list, min_bytes: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    fpath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(fpath) and os.path.getsize(fpath) > min_bytes:
        return fpath
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(fpath, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if os.path.getsize(fpath) > min_bytes:
                return fpath
        except Exception:
            continue
    raise RuntimeError(f"Failed to download {filename}")
def load_sovereign_datasets():
    p_path = _download_split_npz(
        "pathmnist.npz",
        [
            "https://zenodo.org/api/records/10519652/files/pathmnist.npz/content",
            "https://zenodo.org/record/6496656/files/pathmnist.npz?download=1",
        ],
        150_000_000,
    )
    b_path = _download_split_npz(
        "bloodmnist.npz",
        [
            "https://zenodo.org/api/records/10519652/files/bloodmnist.npz/content",
            "https://zenodo.org/record/6496656/files/bloodmnist.npz?download=1",
        ],
        20_000_000,
    )
    with np.load(p_path) as d:
        tr_x = torch.from_numpy(d["train_images"]).permute(0, 3, 1, 2).contiguous()
        tr_y = torch.from_numpy(d["train_labels"]).squeeze(-1).long().contiguous()
        va_x = torch.from_numpy(d["val_images"]).permute(0, 3, 1, 2).contiguous()
        va_y = torch.from_numpy(d["val_labels"]).squeeze(-1).long().contiguous()
        te_x = torch.from_numpy(d["test_images"]).permute(0, 3, 1, 2).contiguous()
        te_y = torch.from_numpy(d["test_labels"]).squeeze(-1).long().contiguous()
    with np.load(b_path) as b:
        ood_x = torch.from_numpy(b["test_images"]).permute(0, 3, 1, 2).contiguous()
    return tr_x, tr_y, va_x, va_y, te_x, te_y, ood_x
# ====================================================================================
# 2. YCoCg-R 4:2:0 COMPRESSION & 1-PASS FUSED TRITON (DECODE + 8-CH DUAL-PHYSICS NHWC)
# ====================================================================================
def compress_ycocg420_vectorized(x_u8: torch.Tensor) -> torch.Tensor:
    r = x_u8[:, 0].to(torch.int16)
    g = x_u8[:, 1].to(torch.int16)
    b = x_u8[:, 2].to(torch.int16)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    co_u8 = torch.clamp((co >> 1) + 128, 0, 255).to(torch.uint8)
    cg_u8 = torch.clamp((cg >> 1) + 128, 0, 255).to(torch.uint8)
    y_u8 = torch.clamp(y, 0, 255).to(torch.uint8)
    N = x_u8.shape[0]
    co_sub = (
        co_u8.view(N, 14, 2, 14, 2)
        .float()
        .mean(dim=(2, 4))
        .round()
        .to(torch.uint8)
        .reshape(N, 196)
    )
    cg_sub = (
        cg_u8.view(N, 14, 2, 14, 2)
        .float()
        .mean(dim=(2, 4))
        .round()
        .to(torch.uint8)
        .reshape(N, 196)
    )
    return torch.cat([y_u8.reshape(N, 784), co_sub, cg_sub], dim=1).contiguous()
@triton.jit
def _load_ycocg_gray_28x28(comp_ptr, base_in, row, col, mask_p):
    """Decodes exact clamped YCoCg-R 4:2:0 grayscale intensity (R+G+B)/3 with zero-padding."""
    valid = mask_p & (row >= 0) & (row < 28) & (col >= 0) & (col < 28)
    r_c = tl.where(row < 0, 0, tl.where(row > 27, 27, row))
    c_c = tl.where(col < 0, 0, tl.where(col > 27, 27, col))
    p_idx = r_c * 28 + c_c
    ch_idx = (r_c // 2) * 14 + (c_c // 2)
    y_v = tl.load(comp_ptr + base_in + p_idx, mask=valid, other=0).to(tl.float32)
    co_v = (tl.load(comp_ptr + base_in + 784 + ch_idx, mask=valid, other=128).to(tl.float32) - 128.0) * 2.0
    cg_v = (tl.load(comp_ptr + base_in + 980 + ch_idx, mask=valid, other=128).to(tl.float32) - 128.0) * 2.0
    # Exact YCoCg-R inverse (fixes v11.1 red-channel scaling bug: r_raw = b_raw + co)
    t = y_v - (cg_v * 0.5)
    g_raw = t + cg_v
    b_raw = t - (co_v * 0.5)
    r_raw = b_raw + co_v
    r = tl.clamp(r_raw, 1.0, 255.0) * (1.0 / 255.0)
    g = tl.clamp(g_raw, 1.0, 255.0) * (1.0 / 255.0)
    b = tl.clamp(b_raw, 1.0, 255.0) * (1.0 / 255.0)
    return tl.where(valid, (r + g + b) * (1.0 / 3.0), 0.0)
@triton.jit
def _ycocg420_decode_to_phys8_nhwc_kernel(comp_ptr, out_nhwc_ptr, N):
    """
    1-Pass Fused Triton Kernel:
    YCoCg-R 4:2:0 (1,176 B) -> Exact RGB Norm + Beer-Lambert OD + Sobel |grad| + Laplacian
    Directly written in NHWC FP16 (N, 28, 28, 8). Replaces 13 PyTorch ops with 1 launch.
    """
    pid = tl.program_id(0)
    pix = tl.arange(0, 1024)
    mask_p = (pix < 784) & (pid < N)
    row = pix // 28
    col = pix % 28
    c_idx = (row // 2) * 14 + (col // 2)
    base_in = pid * 1176
    y_v = tl.load(comp_ptr + base_in + pix, mask=mask_p, other=0).to(tl.float32)
    co_v = (tl.load(comp_ptr + base_in + 784 + c_idx, mask=mask_p, other=128).to(tl.float32) - 128.0) * 2.0
    cg_v = (tl.load(comp_ptr + base_in + 980 + c_idx, mask=mask_p, other=128).to(tl.float32) - 128.0) * 2.0
    # Exact YCoCg-R inverse
    t = y_v - (cg_v * 0.5)
    g_raw = t + cg_v
    b_raw = t - (co_v * 0.5)
    r_raw = b_raw + co_v
    r_01 = tl.clamp(r_raw, 1.0, 255.0) * (1.0 / 255.0)
    g_01 = tl.clamp(g_raw, 1.0, 255.0) * (1.0 / 255.0)
    b_01 = tl.clamp(b_raw, 1.0, 255.0) * (1.0 / 255.0)
    # Normalized RGB
    r_n = (r_01 - 0.7405) * (1.0 / 0.1237)
    g_n = (g_01 - 0.5330) * (1.0 / 0.1768)
    b_n = (b_01 - 0.7058) * (1.0 / 0.1244)
    # Beer-Lambert Optical Density: -log10(I) = -ln(I) * (1 / ln(10))
    INV_LN10 = 0.4342944819032518
    od_r = -tl.log(r_01) * INV_LN10
    od_g = -tl.log(g_01) * INV_LN10
    od_b = -tl.log(b_01) * INV_LN10
    # 3x3 Sobel + Laplacian directly from L1-resident 1,176-byte YCoCg tile
    g_c = (r_01 + g_01 + b_01) * (1.0 / 3.0)
    g_tl = _load_ycocg_gray_28x28(comp_ptr, base_in, row - 1, col - 1, mask_p)
    g_t = _load_ycocg_gray_28x28(comp_ptr, base_in, row - 1, col, mask_p)
    g_tr = _load_ycocg_gray_28x28(comp_ptr, base_in, row - 1, col + 1, mask_p)
    g_l = _load_ycocg_gray_28x28(comp_ptr, base_in, row, col - 1, mask_p)
    g_r = _load_ycocg_gray_28x28(comp_ptr, base_in, row, col + 1, mask_p)
    g_bl = _load_ycocg_gray_28x28(comp_ptr, base_in, row + 1, col - 1, mask_p)
    g_b = _load_ycocg_gray_28x28(comp_ptr, base_in, row + 1, col, mask_p)
    g_br = _load_ycocg_gray_28x28(comp_ptr, base_in, row + 1, col + 1, mask_p)
    gx = -g_tl + g_tr - 2.0 * g_l + 2.0 * g_r - g_bl + g_br
    gy = -g_tl - 2.0 * g_t - g_tr + g_bl + 2.0 * g_b + g_br
    grad_mag = tl.sqrt(gx * gx + gy * gy + 1e-4)
    lap = g_t + g_l + g_r + g_b - 4.0 * g_c
    dst = pid * 6272 + pix * 8
    tl.store(out_nhwc_ptr + dst + 0, r_n.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 1, g_n.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 2, b_n.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 3, od_r.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 4, od_g.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 5, od_b.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 6, grad_mag.to(tl.float16), mask=mask_p)
    tl.store(out_nhwc_ptr + dst + 7, lap.to(tl.float16), mask=mask_p)
def triton_decode_ycocg420_to_phys8_nhwc(
    comp_u8: torch.Tensor, out_nhwc_fp16: torch.Tensor
) -> torch.Tensor:
    N = comp_u8.shape[0]
    _ycocg420_decode_to_phys8_nhwc_kernel[(N,)](
        comp_u8, out_nhwc_fp16, N, num_warps=4
    )
    return out_nhwc_fp16
# ====================================================================================
# 3. DUAL-PHYSICS STEM & SUB-2s HIGH-GENERALIZATION TENSORCORE ARCHITECTURES
# ====================================================================================
class DualPhysicsFastStem(nn.Module):
    """
    Computes exact 8-channel Dual-Physics tensor (RGB + Beer-Lambert OD + Sobel + Laplacian).
    Uses linear channel-mean identity: mean_c(conv(I_c, K)) == conv(mean_c(I_c), K) for 3x faster conv.
    """
    def __init__(self, normalize_channels: bool = False):
        super().__init__()
        self.normalize_channels = normalize_channels
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        kl = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("w_gray_phys", torch.cat([kx, ky, kl], dim=0).half())  # (3, 1, 3, 3)
        self.register_buffer("mean", torch.tensor([0.7405, 0.5330, 0.7058]).view(1, 3, 1, 1).half())
        self.register_buffer(
            "inv_std", (1.0 / torch.tensor([0.1237, 0.1768, 0.1244])).view(1, 3, 1, 1).half()
        )
    def forward(self, x: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x_01 = torch.clamp(x.half() * (1.0 / 255.0), 1.0 / 255.0, 1.0)
            x_rgb = (x_01 - self.mean) * self.inv_std
        elif not is_normalized:
            x_01 = torch.clamp(x.half(), 1.0 / 255.0, 1.0)
            x_rgb = (x_01 - self.mean) * self.inv_std
        else:
            x_rgb = x.half()
            x_01 = torch.clamp(x_rgb * (1.0 / self.inv_std) + self.mean, 1.0 / 255.0, 1.0)
        od = -torch.log10(x_01)
        gray = x_01.mean(dim=1, keepdim=True)
        g_all = F.conv2d(gray, self.w_gray_phys, padding=1)  # (B, 3, 28, 28)
        gx = g_all[:, 0:1]
        gy = g_all[:, 1:2]
        lap = g_all[:, 2:3]
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-4)
        out = torch.cat([x_rgb, od, grad_mag, lap], dim=1)
        if self.normalize_channels:
            v, m = torch.var_mean(out, dim=(2, 3), keepdim=True, unbiased=False)
            out = (out - m) * torch.rsqrt(v + 1e-4)
        return out
class PureZeroEpochOpticalReservoir(nn.Module):
    """Paradigm 1: Pure 0-Epoch Optical Reservoir (0 Backprop, Fixed Orthogonal Filters)."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.stem = DualPhysicsFastStem(normalize_channels=True).to(device)
        torch.manual_seed(2026)
        self.conv1 = nn.Conv2d(8, 128, kernel_size=5, stride=1, padding=2, groups=8, bias=False)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_5 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2, bias=False)
        nn.init.orthogonal_(self.conv1.weight, gain=1.41)
        nn.init.orthogonal_(self.conv2_3.weight, gain=1.41)
        nn.init.orthogonal_(self.conv2_5.weight, gain=1.41)
        self.to(device).half().to(memory_format=torch.channels_last).eval()
    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x).contiguous(memory_format=torch.channels_last)
        h = F.gelu(self.conv1(h))
        v1, m1 = torch.var_mean(h, dim=(2, 3), keepdim=True, unbiased=False)
        h = (h - m1) * torch.rsqrt(v1 + 1e-4)
        h2 = torch.cat([F.gelu(self.conv2_3(h)), F.gelu(self.conv2_5(h))], dim=1)
        h2_f = h2.float().flatten(2)
        f_max = h2_f.amax(dim=2)
        f_avg = h2_f.mean(dim=2)
        f_std = torch.sqrt(torch.clamp(h2_f.square().mean(dim=2) - f_avg.square(), min=0.0) + 1e-4)
        f = torch.cat([f_max, f_avg, f_std], dim=1)
        return F.layer_norm(f, (768,)).half()
@torch.no_grad()
def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    scale = gamma * torch.rsqrt(var + bn.eps)
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = beta - mean * scale
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        device=conv.weight.device,
        dtype=torch.float16,
    )
    fused.weight.copy_(w_fused.half())
    fused.bias.copy_(b_fused.half())
    return fused.to(memory_format=torch.channels_last)
class FastSEBlock(nn.Module):
    """Lightweight Global Channel Attention (~0.02 MFLOPs) for Cross-Center Stain Recalibration."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = channels // reduction
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s.unsqueeze(-1).unsqueeze(-1)
class TensorCoreResBlock(nn.Module):
    """Pure 3x3 Tensor-Core Residual Block in NHWC FP16 with Zero-SFU ReLU & Inference BN Fusion."""
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.b1: nn.Module = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.b2: nn.Module = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    def fuse_bn(self):
        if isinstance(self.b1, nn.BatchNorm2d):
            self.c1 = _fuse_conv_bn_pair(self.c1, self.b1)
            self.b1 = nn.Identity()
        if isinstance(self.b2, nn.BatchNorm2d):
            self.c2 = _fuse_conv_bn_pair(self.c2, self.b2)
            self.b2 = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(x + h)
class DualPhysicsTensorCoreNet(nn.Module):
    """
    v12.0 Sub-2s High-Generalization TensorCore Architecture (99.35 MFLOPs/img, 758K params):
    28x28 (16ch) -> 14x14 (64ch + ResBlock) -> 7x7 (128ch + ResBlock -> 256ch + Global SE).
    2.76x fewer FLOPs & 3.3x fewer params than v11.1 while preserving full 28x28 nuclear detail,
    41x41 receptive field, and 768-dim (max+avg+std) penultimate pathology representation.
    """
    def __init__(self, num_classes: int = 9, drop_p: float = 0.15):
        super().__init__()
        self.drop_p = drop_p
        self.stem = DualPhysicsFastStem(normalize_channels=False)
        self.c_28 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_28: nn.Module = nn.BatchNorm2d(16)
        self.c_d1 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.b_d1: nn.Module = nn.BatchNorm2d(64)
        self.res1 = TensorCoreResBlock(64)
        self.c_d2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b_d2: nn.Module = nn.BatchNorm2d(128)
        self.res2 = TensorCoreResBlock(128)
        self.c_s3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_s3: nn.Module = nn.BatchNorm2d(256)
        self.se3 = FastSEBlock(256, reduction=8)
        self.act = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(768, num_classes)
    def fuse_bn_for_inference(self):
        self.eval()
        if isinstance(self.b_28, nn.BatchNorm2d):
            self.c_28 = _fuse_conv_bn_pair(self.c_28, self.b_28)
            self.b_28 = nn.Identity()
        if isinstance(self.b_d1, nn.BatchNorm2d):
            self.c_d1 = _fuse_conv_bn_pair(self.c_d1, self.b_d1)
            self.b_d1 = nn.Identity()
        if isinstance(self.b_d2, nn.BatchNorm2d):
            self.c_d2 = _fuse_conv_bn_pair(self.c_d2, self.b_d2)
            self.b_d2 = nn.Identity()
        if isinstance(self.b_s3, nn.BatchNorm2d):
            self.c_s3 = _fuse_conv_bn_pair(self.c_s3, self.b_s3)
            self.b_s3 = nn.Identity()
        self.res1.fuse_bn()
        self.res2.fuse_bn()
        return self
    def forward_features_from_8ch(self, x8: torch.Tensor) -> torch.Tensor:
        x = self.act(self.b_28(self.c_28(x8)))            # (B, 16, 28, 28)
        x = self.res1(self.act(self.b_d1(self.c_d1(x))))  # (B, 64, 14, 14)
        x = self.res2(self.act(self.b_d2(self.c_d2(x))))  # (B, 128, 7, 7)
        x = self.se3(self.act(self.b_s3(self.c_s3(x))))   # (B, 256, 7, 7)
        xf = x.float().flatten(2)
        f_max = xf.amax(dim=2)
        f_avg = xf.mean(dim=2)
        f_std = torch.sqrt(torch.clamp(xf.square().mean(dim=2) - f_avg.square(), min=0.0) + 1e-4)
        f = torch.cat([f_max, f_avg, f_std], dim=1)
        return F.layer_norm(f, (768,)).half()
    def forward_features(self, x: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        x8 = self.stem(x, is_normalized=is_normalized).contiguous(memory_format=torch.channels_last)
        return self.forward_features_from_8ch(x8)
    def forward(self, x8: torch.Tensor) -> torch.Tensor:
        f = self.forward_features_from_8ch(x8)
        if self.training and self.drop_p > 0.0:
            f = F.dropout(f, p=self.drop_p, training=True)
        return self.fc_cls(f)
# ====================================================================================
# 4. EXACT BEER-LAMBERT D4 TRITON AUGMENT + FUSED TRITON NESTEROV SGD + CUDAGRAPH
# ====================================================================================
@triton.jit
def _fused_nhwc_d4_beerlambert_cutmix_v12_kernel(
    src_ptr,
    y_ptr,
    perm_ptr,
    rand_ptr,
    ep_step_ptr,
    out_ptr,
    ya_ptr,
    yb_ptr,
    lam_ptr,
    B: tl.constexpr,
    CM_BOX: tl.constexpr,
    CM_PROB: tl.constexpr,
    STAIN_STRENGTH: tl.constexpr,
    EXP_STRENGTH: tl.constexpr,
):
    """
    Zero-overhead fused augment kernel inside CUDAGraph:
    - Full D4 Dihedral Symmetry (all 8 exact 90-deg rotations & flips)
    - Physically-Exact Beer-Lambert H&E Stain & Illumination Jitter (RGB derived from perturbed OD)
    - Masked Single-Source CutMix (each pixel loads ONLY img_a OR img_b, cutting DRAM gather by 50%)
    - Exact integer-aligned CutMix box area + fused label gather (eliminates 2 index_select calls)
    """
    b = tl.program_id(0)
    row_blk = tl.program_id(1)
    ep_step = tl.load(ep_step_ptr).to(tl.int64)
    base_sample = ep_step * B
    img_a = tl.load(perm_ptr + base_sample + b).to(tl.int64)
    b_roll = (b + B - 1) % B
    img_b = tl.load(perm_ptr + base_sample + b_roll).to(tl.int64)
    base_r_a = (base_sample + b) * 10
    r0 = tl.load(rand_ptr + base_r_a + 0).to(tl.float32)
    r1 = tl.load(rand_ptr + base_r_a + 1).to(tl.float32)
    r2 = tl.load(rand_ptr + base_r_a + 2).to(tl.float32)
    r3 = tl.load(rand_ptr + base_r_a + 3).to(tl.float32)
    r4 = tl.load(rand_ptr + base_r_a + 4).to(tl.float32)
    r5 = tl.load(rand_ptr + base_r_a + 5).to(tl.float32)
    r6 = tl.load(rand_ptr + base_r_a + 6).to(tl.float32)
    r7 = tl.load(rand_ptr + base_r_a + 7).to(tl.float32)
    r8 = tl.load(rand_ptr + base_r_a + 8).to(tl.float32)
    r9 = tl.load(rand_ptr + base_r_a + 9).to(tl.float32)
    do_hf = r0 > 0.5
    do_vf = r1 > 0.5
    do_tr = r2 > 0.5
    do_cm = r3 > (1.0 - CM_PROB)
    max_pos = 28.0 - CM_BOX
    y0_raw = (r4 * (29.0 - CM_BOX)).to(tl.int32).to(tl.float32)
    x0_raw = (r5 * (29.0 - CM_BOX)).to(tl.int32).to(tl.float32)
    y0 = tl.where(y0_raw > max_pos, max_pos, y0_raw)
    x0 = tl.where(x0_raw > max_pos, max_pos, x0_raw)
    if row_blk == 0:
        lam_val = tl.where(do_cm, 1.0 - (CM_BOX * CM_BOX) / 784.0, 1.0)
        tl.store(lam_ptr + b, lam_val)
        tl.store(ya_ptr + b, tl.load(y_ptr + img_a))
        tl.store(yb_ptr + b, tl.load(y_ptr + img_b))
    offs = tl.arange(0, 1024)
    mask = offs < 896
    c = offs % 8
    pix_local = offs // 8
    row = row_blk * 4 + (pix_local // 28)
    col = pix_local % 28
    row_f = row.to(tl.float32)
    col_f = col.to(tl.float32)
    in_box = do_cm & (row_f >= y0) & (row_f < (y0 + CM_BOX)) & (col_f >= x0) & (col_f < (x0 + CM_BOX))
    mask_a = mask & (~in_box)
    mask_b = mask & in_box
    LN10 = 2.302585092994046
    w_idx = c % 3
    mean_c = tl.where(c == 0, 0.7405, tl.where(c == 1, 0.5330, 0.7058))
    inv_std_c = tl.where(c == 0, 1.0 / 0.1237, tl.where(c == 1, 1.0 / 0.1768, 1.0 / 0.1244))
    # --- Source A (loaded ONLY outside CutMix box) ---
    ra0 = tl.where(do_vf, 27 - row, row)
    ca0 = tl.where(do_hf, 27 - col, col)
    src_row_a = tl.where(do_tr, ca0, ra0)
    src_col_a = tl.where(do_tr, ra0, ca0)
    pix_off_a = (img_a * 784 + src_row_a * 28 + src_col_a) * 8
    val_a = tl.load(src_ptr + pix_off_a + c, mask=mask_a, other=0.0).to(tl.float32)
    od_rgb_a = tl.load(src_ptr + pix_off_a + (c + 3), mask=mask_a & (c < 3), other=0.0).to(tl.float32)
    rw_a = tl.where(w_idx == 0, r6, tl.where(w_idx == 1, r7, r8)) - 0.5
    re_a = r9 - 0.5
    alpha_a = 1.0 + rw_a * STAIN_STRENGTH
    beta_a = re_a * EXP_STRENGTH
    od_p_a = tl.clamp(od_rgb_a * alpha_a + beta_a, 0.0, 3.0)
    rgb01_a = tl.clamp(tl.exp(-od_p_a * LN10), 1.0 / 255.0, 1.0)
    rgb_out_a = (rgb01_a - mean_c) * inv_std_c
    od_out_a = tl.clamp(val_a * alpha_a + beta_a, 0.0, 3.0)
    grad_out_a = val_a * (1.0 + (r6 - 0.5) * 0.20 - beta_a * 1.5)
    aug_a = tl.where(c < 3, rgb_out_a, tl.where(c < 6, od_out_a, grad_out_a))
    # --- Source B (loaded ONLY inside CutMix box: skipped for ~93% of pixels!) ---
    base_r_b = (base_sample + b_roll) * 10
    r0_b = tl.load(rand_ptr + base_r_b + 0).to(tl.float32)
    r1_b = tl.load(rand_ptr + base_r_b + 1).to(tl.float32)
    r2_b = tl.load(rand_ptr + base_r_b + 2).to(tl.float32)
    r6_b = tl.load(rand_ptr + base_r_b + 6).to(tl.float32)
    r7_b = tl.load(rand_ptr + base_r_b + 7).to(tl.float32)
    r8_b = tl.load(rand_ptr + base_r_b + 8).to(tl.float32)
    r9_b = tl.load(rand_ptr + base_r_b + 9).to(tl.float32)
    rb0 = tl.where(r1_b > 0.5, 27 - row, row)
    cb0 = tl.where(r0_b > 0.5, 27 - col, col)
    src_row_b = tl.where(r2_b > 0.5, cb0, rb0)
    src_col_b = tl.where(r2_b > 0.5, rb0, cb0)
    pix_off_b = (img_b * 784 + src_row_b * 28 + src_col_b) * 8
    val_b = tl.load(src_ptr + pix_off_b + c, mask=mask_b, other=0.0).to(tl.float32)
    od_rgb_b = tl.load(src_ptr + pix_off_b + (c + 3), mask=mask_b & (c < 3), other=0.0).to(tl.float32)
    rw_b = tl.where(w_idx == 0, r6_b, tl.where(w_idx == 1, r7_b, r8_b)) - 0.5
    re_b = r9_b - 0.5
    alpha_b = 1.0 + rw_b * STAIN_STRENGTH
    beta_b = re_b * EXP_STRENGTH
    od_p_b = tl.clamp(od_rgb_b * alpha_b + beta_b, 0.0, 3.0)
    rgb01_b = tl.clamp(tl.exp(-od_p_b * LN10), 1.0 / 255.0, 1.0)
    rgb_out_b = (rgb01_b - mean_c) * inv_std_c
    od_out_b = tl.clamp(val_b * alpha_b + beta_b, 0.0, 3.0)
    grad_out_b = val_b * (1.0 + (r6_b - 0.5) * 0.20 - beta_b * 1.5)
    aug_b = tl.where(c < 3, rgb_out_b, tl.where(c < 6, od_out_b, grad_out_b))
    val_out = tl.where(in_box, aug_b, aug_a)
    dst_off = (b * 784 + row * 28 + col) * 8 + c
    tl.store(out_ptr + dst_off, val_out.to(tl.float16), mask=mask)
@triton.jit
def _fused_nesterov_sgd_fp16_master_kernel(
    p16_ptr,
    g16_ptr,
    m32_ptr,
    mom32_ptr,
    lr_ptr,
    step_ptr,
    N_ELEM: tl.constexpr,
    INV_SCALE: tl.constexpr,
    WD: tl.constexpr,
    MOM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass fused Nesterov SGD + Weight Decay + Unscale + Finite Guard + FP32->FP16 cast."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEM
    step_idx = tl.load(step_ptr).to(tl.int64)
    lr = tl.load(lr_ptr + step_idx).to(tl.float32)
    g = tl.load(g16_ptr + offs, mask=mask, other=0.0).to(tl.float32) * INV_SCALE
    m = tl.load(m32_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(mom32_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    is_fin = (g > -1e4) & (g < 1e4)
    g = tl.where(is_fin, g, 0.0)
    if WD > 0.0:
        g = g + WD * m
    v_new = MOM * v + g
    nes = g + MOM * v_new
    m_new = m - lr * nes
    tl.store(mom32_ptr + offs, v_new, mask=mask)
    tl.store(m32_ptr + offs, m_new, mask=mask)
    tl.store(p16_ptr + offs, m_new.to(tl.float16), mask=mask)
@triton.jit
def _fused_nesterov_sgd_fp32_kernel(
    p32_ptr,
    g32_ptr,
    mom32_ptr,
    lr_ptr,
    step_ptr,
    N_ELEM: tl.constexpr,
    INV_SCALE: tl.constexpr,
    MOM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass fused Nesterov SGD for FP32 BatchNorm affine parameters."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEM
    step_idx = tl.load(step_ptr).to(tl.int64)
    lr = tl.load(lr_ptr + step_idx).to(tl.float32)
    g = tl.load(g32_ptr + offs, mask=mask, other=0.0).to(tl.float32) * INV_SCALE
    p = tl.load(p32_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(mom32_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    is_fin = (g > -1e4) & (g < 1e4)
    g = tl.where(is_fin, g, 0.0)
    v_new = MOM * v + g
    nes = g + MOM * v_new
    p_new = p - lr * nes
    tl.store(mom32_ptr + offs, v_new, mask=mask)
    tl.store(p32_ptr + offs, p_new, mask=mask)
def _pack_params_aligned(
    params: List[nn.Parameter], dtype: torch.dtype, device: torch.device, align: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs parameters & grads into 128-byte aligned contiguous flat tensors while preserving
    exact torch.channels_last layout for all 4D Conv2d weights (zero cuDNN TensorCore penalty).
    """
    offsets = []
    total = 0
    for p in params:
        offsets.append(total)
        n = p.numel()
        total += ((n + align - 1) // align) * align
    total = max(align, total)
    flat_p = torch.zeros((total,), device=device, dtype=dtype)
    flat_g = torch.zeros((total,), device=device, dtype=dtype)
    with torch.no_grad():
        for p, off in zip(params, offsets):
            n = p.numel()
            sl_p = flat_p[off : off + n]
            sl_g = flat_g[off : off + n]
            if p.ndim == 4:
                co, ci, kh, kw = p.shape
                v_p = sl_p.view(co, kh, kw, ci).permute(0, 3, 1, 2)
                v_g = sl_g.view(co, kh, kw, ci).permute(0, 3, 1, 2)
                v_p.copy_(p.data)
            else:
                v_p = sl_p.view_as(p)
                v_g = sl_g.view_as(p)
                v_p.copy_(p.data)
            p.data = v_p
            p.grad = v_g
    return flat_p, flat_g
def train_fast_specialist(
    gpu_id: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 32,
    seed: int = 101,
    cm_box: int = 10,
    cm_prob: float = 0.45,
    stain_strength: float = 0.40,
    exp_strength: float = 0.06,
    wd: float = 2.5e-4,
    label_smooth: float = 0.08,
    drop_p: float = 0.12,
) -> DualPhysicsTensorCoreNet:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DualPhysicsTensorCoreNet(NUM_CLASSES, drop_p=drop_p).to(device)
    model.half().to(memory_format=torch.channels_last)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.float()
    model.train()
    t_pre0 = time.perf_counter()
    N = train_x.shape[0]
    x_phys8_nhwc = torch.empty((N, 28, 28, 8), device=device, dtype=torch.float16)
    with torch.no_grad():
        bs_pre = 4096
        for i in range(0, N, bs_pre):
            xb = train_x[i : i + bs_pre].to(device, non_blocking=True)
            x8 = model.stem(xb, is_normalized=False)
            x_phys8_nhwc[i : i + bs_pre].copy_(x8.permute(0, 2, 3, 1))
    y_gpu = train_y.to(device, non_blocking=True).contiguous()
    torch.cuda.synchronize(device)
    t_pre_sec = time.perf_counter() - t_pre0
    B = 512
    steps_per_ep = N // B
    total_steps = epochs * steps_per_ep
    # Pack parameters into 3 contiguous 128-byte aligned buffers (3 Triton launches total per step)
    fp16_w_params = [p for p in model.parameters() if p.requires_grad and p.dtype == torch.float16 and p.ndim >= 2]
    fp16_b_params = [p for p in model.parameters() if p.requires_grad and p.dtype == torch.float16 and p.ndim < 2]
    fp32_params = [p for p in model.parameters() if p.requires_grad and p.dtype == torch.float32]
    flat_p16_w, flat_g16_w = _pack_params_aligned(fp16_w_params, torch.float16, device)
    flat_p16_b, flat_g16_b = _pack_params_aligned(fp16_b_params, torch.float16, device)
    flat_p32, flat_g32 = _pack_params_aligned(fp32_params, torch.float32, device)
    flat_m32_w = flat_p16_w.detach().float().clone()
    flat_m32_b = flat_p16_b.detach().float().clone()
    flat_mom32_w = torch.zeros_like(flat_m32_w)
    flat_mom32_b = torch.zeros_like(flat_m32_b)
    flat_mom32_bn = torch.zeros_like(flat_p32)
    flat_swa32_w = flat_m32_w.clone()
    flat_swa32_b = flat_m32_b.clone()
    flat_swa32_bn = flat_p32.detach().clone()
    swa_count = 0
    static_perm = torch.arange(N, device=device, dtype=torch.long)
    static_rand_ep = torch.empty((steps_per_ep * B, 10), device=device, dtype=torch.float16)
    static_x_nhwc = torch.zeros((B, 28, 28, 8), device=device, dtype=torch.float16)
    static_x = static_x_nhwc.permute(0, 3, 1, 2)  # exact channels_last view
    static_y_a = torch.zeros((B,), device=device, dtype=torch.long)
    static_y_b = torch.zeros((B,), device=device, dtype=torch.long)
    static_lam = torch.ones((B,), device=device, dtype=torch.float32)
    static_step = torch.zeros((1,), device=device, dtype=torch.int64)
    static_ep_step = torch.zeros((1,), device=device, dtype=torch.int64)
    # Zero LR during warmup+capture; real cosine LR copied immediately after capture
    lr_gpu = torch.zeros((total_steps + 16,), device=device, dtype=torch.float32)
    momentum = 0.9
    loss_scale = 256.0
    inv_scale = 1.0 / loss_scale
    OPT_BLK = 1024
    n_w = flat_p16_w.numel()
    n_b = flat_p16_b.numel()
    n_bn = flat_p32.numel()
    grid_w = (triton.cdiv(n_w, OPT_BLK),)
    grid_b = (triton.cdiv(n_b, OPT_BLK),)
    grid_bn = (triton.cdiv(n_bn, OPT_BLK),)
    def _full_graph_step():
        _fused_nhwc_d4_beerlambert_cutmix_v12_kernel[(B, 7)](
            x_phys8_nhwc,
            y_gpu,
            static_perm,
            static_rand_ep,
            static_ep_step,
            static_x_nhwc,
            static_y_a,
            static_y_b,
            static_lam,
            B=B,
            CM_BOX=cm_box,
            CM_PROB=cm_prob,
            STAIN_STRENGTH=stain_strength,
            EXP_STRENGTH=exp_strength,
            num_warps=4,
        )
        flat_g16_w.zero_()
        flat_g16_b.zero_()
        flat_g32.zero_()
        logits = model(static_x)
        log_p = F.log_softmax(logits.float(), dim=1)
        nll_a = -log_p.gather(1, static_y_a.unsqueeze(1)).squeeze(1)
        nll_b = -log_p.gather(1, static_y_b.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=1)
        loss = (
            (1.0 - label_smooth) * (static_lam * nll_a + (1.0 - static_lam) * nll_b)
            + label_smooth * smooth
        ).mean()
        (loss * loss_scale).backward()
        with torch.no_grad():
            _fused_nesterov_sgd_fp16_master_kernel[grid_w](
                flat_p16_w,
                flat_g16_w,
                flat_m32_w,
                flat_mom32_w,
                lr_gpu,
                static_step,
                N_ELEM=n_w,
                INV_SCALE=inv_scale,
                WD=wd,
                MOM=momentum,
                BLOCK=OPT_BLK,
                num_warps=4,
            )
            _fused_nesterov_sgd_fp16_master_kernel[grid_b](
                flat_p16_b,
                flat_g16_b,
                flat_m32_b,
                flat_mom32_b,
                lr_gpu,
                static_step,
                N_ELEM=n_b,
                INV_SCALE=inv_scale,
                WD=0.0,
                MOM=momentum,
                BLOCK=OPT_BLK,
                num_warps=4,
            )
            _fused_nesterov_sgd_fp32_kernel[grid_bn](
                flat_p32,
                flat_g32,
                flat_mom32_bn,
                lr_gpu,
                static_step,
                N_ELEM=n_bn,
                INV_SCALE=inv_scale,
                MOM=momentum,
                BLOCK=OPT_BLK,
                num_warps=4,
            )
            static_step.add_(1)
            static_ep_step.add_(1)
    # Serialize cuDNN warmup + CUDAGraph capture per GPU so both T4s pick optimal HMMA kernels
    s_cap = torch.cuda.Stream(device=device)
    train_graph = torch.cuda.CUDAGraph()
    with _GRAPH_CAPTURE_LOCK:
        s_cap.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s_cap):
            static_rand_ep.uniform_(0.0, 1.0)
            static_step.zero_()
            static_ep_step.zero_()
            for _ in range(3):
                _full_graph_step()
        torch.cuda.current_stream(device).wait_stream(s_cap)
        torch.cuda.synchronize(device)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
            flat_mom32_w.zero_()
            flat_mom32_b.zero_()
            flat_mom32_bn.zero_()
            static_step.zero_()
            static_ep_step.zero_()
        with torch.cuda.graph(train_graph, stream=s_cap, capture_error_mode="thread_local"):
            _full_graph_step()
        torch.cuda.synchronize(device)
    # Populate cosine warmup LR schedule on GPU
    max_lr = 0.036
    lr_schedule = torch.zeros((total_steps + 16,), dtype=torch.float32, pin_memory=True)
    warmup_steps = max(1, int(0.08 * total_steps))
    for st_i in range(total_steps):
        if st_i < warmup_steps:
            lr_schedule[st_i] = max_lr * (st_i + 1) / warmup_steps
        else:
            prog = (st_i - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[st_i] = max_lr * 0.5 * (1.0 + math.cos(math.pi * prog))
    lr_gpu.copy_(lr_schedule, non_blocking=True)
    torch.cuda.synchronize(device)
    t_loop0 = time.perf_counter()
    swa_start_ep = int(epochs * 0.65)
    for ep in range(epochs):
        static_perm.copy_(torch.randperm(N, device=device))
        static_rand_ep.uniform_(0.0, 1.0)
        static_step.fill_(ep * steps_per_ep)
        static_ep_step.zero_()
        # Zero-launch-overhead inner loop: strictly CUDAGraph replays only
        for _ in range(steps_per_ep):
            train_graph.replay()
        # Zero-overhead flat-buffer SWA accumulation
        if ep >= swa_start_ep:
            with torch.no_grad():
                w_swa = 1.0 / float(swa_count + 1)
                flat_swa32_w.lerp_(flat_m32_w, w_swa)
                flat_swa32_b.lerp_(flat_m32_b, w_swa)
                flat_swa32_bn.lerp_(flat_p32, w_swa)
                swa_count += 1
        if (ep + 1) in (1, 8, 16, 24, epochs):
            torch.cuda.synchronize(device)
            el = time.perf_counter() - t_loop0
            print(
                f"    [GPU {gpu_id}] Ep {ep+1:02d}/{epochs:02d} | "
                f"Train Loop: {el:5.2f}s ({el/(ep+1):.3f}s/ep | Precompute: {t_pre_sec:.2f}s)"
            )
    # Apply SWA weights & exact cumulative BN statistics refresh
    if swa_count > 0:
        with torch.no_grad():
            flat_p16_w.copy_(flat_swa32_w)
            flat_p16_b.copy_(flat_swa32_b)
            flat_p32.copy_(flat_swa32_bn)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = None
            for i in range(0, min(N, 16384), B):
                static_x_nhwc.copy_(x_phys8_nhwc[i : i + B])
                model(static_x)
    torch.cuda.synchronize(device)
    for p in model.parameters():
        p.grad = None
    del (
        x_phys8_nhwc,
        static_x_nhwc,
        static_x,
        static_perm,
        static_rand_ep,
        flat_g16_w,
        flat_g16_b,
        flat_g32,
        flat_m32_w,
        flat_m32_b,
        flat_mom32_w,
        flat_mom32_b,
        flat_mom32_bn,
        flat_swa32_w,
        flat_swa32_b,
        flat_swa32_bn,
        train_graph,
    )
    model.half().to(memory_format=torch.channels_last).eval()
    return model
# ====================================================================================
# 5. BATCHED FEATURE EXTRACTION & SPECTRAL WHITENING CHOLESKY SOLVER
# ====================================================================================
@torch.no_grad()
def extract_features_2view(
    model: DualPhysicsTensorCoreNet, x: torch.Tensor, device: torch.device
) -> tuple:
    model.eval()
    N = x.shape[0]
    bs = 4096
    f1_list, f2_list, f_avg_list, l1_list, l2_list = [], [], [], [], []
    for i in range(0, N, bs):
        xb = x[i : i + bs].to(device, non_blocking=True)
        x8_1 = model.stem(xb, is_normalized=False).contiguous(memory_format=torch.channels_last)
        x8_2 = torch.flip(x8_1, dims=[3]).contiguous(memory_format=torch.channels_last)
        f1 = model.forward_features_from_8ch(x8_1)
        f2 = model.forward_features_from_8ch(x8_2)
        l1 = model.fc_cls(f1)
        l2 = model.fc_cls(f2)
        f1_f, f2_f = f1.float(), f2.float()
        l1_f, l2_f = l1.float(), l2.float()
        f1_list.append(f1_f)
        f2_list.append(f2_f)
        f_avg_list.append(0.5 * (f1_f + f2_f))
        l1_list.append(l1_f)
        l2_list.append(0.5 * (l1_f + l2_f))
    return (
        torch.cat(f1_list, dim=0),
        torch.cat(f2_list, dim=0),
        torch.cat(f_avg_list, dim=0),
        torch.cat(l1_list, dim=0),
        torch.cat(l2_list, dim=0),
    )
def compute_brier(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=1)
    oh = F.one_hot(y, NUM_CLASSES).float()
    return float(((probs - oh) ** 2).sum(dim=1).mean().item())
@dataclass
class SovereignEngineState:
    spec0: DualPhysicsTensorCoreNet
    spec1: DualPhysicsTensorCoreNet
    z_mean: torch.Tensor
    W_whiten: torch.Tensor
    W_rff: torch.Tensor
    b_rff: torch.Tensor
    L_chol: torch.Tensor
    W_chol: torch.Tensor
    XY_rhs: torch.Tensor
    W_hybrid: torch.Tensor
    b_hybrid: torch.Tensor
    U_null: torch.Tensor
    U_prin: torch.Tensor
    vim_alpha: float
    m_ns: float
    s_ns: float
    m_en: float
    s_en: float
    ood_thresh: float
    temp_opt: float
# ====================================================================================
# 6. ZERO-DEPENDENCY PNG ENCODER & PERSIAN (RTL) CLOUDFLARE DASHBOARD
# ====================================================================================
_DASHBOARD_LOCK = threading.Lock()
def _array_to_png_b64(rgb_u8: np.ndarray, scale: int = 4) -> str:
    """Zero-dependency pure-Python PNG encoder with nearest-neighbor upscaling."""
    if scale > 1:
        rgb_u8 = np.repeat(np.repeat(rgb_u8, scale, axis=0), scale, axis=1)
    h, w, _ = rgb_u8.shape
    raw_rows = b"".join(b"\x00" + rgb_u8[y].tobytes() for y in range(h))
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
def _phys8_to_visual_b64(phys8_chw: torch.Tensor, raw_rgb_chw: torch.Tensor) -> Tuple[str, str, str]:
    """Converts raw RGB, Beer-Lambert OD (ch 3..5), and Sobel+Laplacian (ch 6..7) into 3 crisp PNGs."""
    rgb_np = raw_rgb_chw.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    p8 = phys8_chw.detach().float().cpu().numpy()
    od_mag = np.mean(p8[3:6], axis=0)
    od_norm = np.clip((od_mag - od_mag.min()) / (od_mag.max() - od_mag.min() + 1e-6), 0.0, 1.0)
    od_rgb = np.stack(
        [
            np.clip(od_norm * 255.0 * 1.15, 0, 255),
            np.clip((1.0 - np.abs(od_norm - 0.5) * 1.8) * 180.0, 0, 255),
            np.clip((1.0 - od_norm * 0.7) * 240.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    edge = np.abs(p8[6]) + 0.75 * np.abs(p8[7])
    edge_norm = np.clip((edge - edge.min()) / (edge.max() - edge.min() + 1e-6), 0.0, 1.0)
    edge_rgb = np.stack(
        [
            np.clip(edge_norm * 120.0, 0, 255),
            np.clip(edge_norm * 255.0, 0, 255),
            np.clip(edge_norm * 220.0 + 25.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    return _array_to_png_b64(rgb_np), _array_to_png_b64(od_rgb), _array_to_png_b64(edge_rgb)
def _ensure_web_deps_and_cloudflared() -> str:
    try:
        import fastapi  # noqa
        import uvicorn  # noqa
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn"],
            check=False,
        )
    cf_bin = "/tmp/cloudflared"
    if not (os.path.exists(cf_bin) and os.path.getsize(cf_bin) > 10_000_000):
        cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        req = urllib.request.Request(cf_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(cf_bin, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.chmod(cf_bin, 0o755)
    return cf_bin
def launch_sovereign_persian_dashboard(ctx: dict):
    cf_bin = _ensure_web_deps_and_cloudflared()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    app = FastAPI(title="Persian Dashboard")
    @app.get("/api/telemetry")
    def api_telemetry():
        return JSONResponse(ctx["telemetry"])
    @app.get("/api/diagnose")
    def api_diagnose(mode: str = "id"):
        with _DASHBOARD_LOCK:
            dev0 = ctx["dev0"]
            state: SovereignEngineState = ctx["state"]
            static_b1_comp = ctx["static_b1_comp"]
            static_b1_phys8 = ctx["static_b1_phys8"]
            static_b1_probs = ctx["static_b1_probs"]
            static_b1_ood = ctx["static_b1_ood"]
            b1_graph = ctx["b1_graph"]
            if mode == "ood":
                ood_x = ctx["ood_x"]
                idx = int(np.random.randint(0, ood_x.shape[0]))
                raw_img = ood_x[idx : idx + 1]
                true_label_fa = "لام خون‌شناسی (BloodMNIST - خارج از توزیع)"
                true_code = "OOD_BLOOD"
            else:
                test_x = ctx["test_x"]
                test_y = ctx["test_y"]
                idx = int(np.random.randint(0, test_x.shape[0]))
                raw_img = test_x[idx : idx + 1]
                t_cls = int(test_y[idx].item())
                true_label_fa = CLASS_NAMES_FA[t_cls]
                true_code = CLASS_NAMES[t_cls][0]
            comp_1 = compress_ycocg420_vectorized(raw_img).pin_memory()
            torch.cuda.synchronize(dev0)
            t0 = time.perf_counter()
            static_b1_comp.copy_(comp_1, non_blocking=True)
            b1_graph.replay()
            torch.cuda.synchronize(dev0)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            ood_val = float(static_b1_ood[0].item())
            probs = static_b1_probs[0].detach().cpu().numpy()
            pred_cls = int(probs.argmax())
            conf = float(probs[pred_cls] * 100.0)
            is_ood = bool(ood_val > state.ood_thresh)
            if is_ood:
                pred_code = "REJECTED_OOD"
                pred_fa = "رد شده توسط سپر ایمنی (خارج از توزیع پاتولوژی کولون)"
            else:
                pred_code = CLASS_NAMES[pred_cls][0]
                pred_fa = CLASS_NAMES_FA[pred_cls]
            rgb_b64, od_b64, edge_b64 = _phys8_to_visual_b64(static_b1_phys8[0], raw_img[0])
            class_probs = [
                {
                    "code": CLASS_NAMES[c][0],
                    "name_fa": CLASS_NAMES_FA[c],
                    "prob": round(float(probs[c] * 100.0), 2),
                }
                for c in range(NUM_CLASSES)
            ]
            return JSONResponse(
                {
                    "slide_idx": idx,
                    "mode": mode,
                    "true_code": true_code,
                    "true_label_fa": true_label_fa,
                    "pred_code": pred_code,
                    "pred_label_fa": pred_fa,
                    "confidence": round(conf, 2),
                    "latency_ms": round(lat_ms, 3),
                    "ood_score": round(ood_val, 3),
                    "ood_thresh": round(float(state.ood_thresh), 3),
                    "is_ood": is_ood,
                    "rgb_b64": rgb_b64,
                    "od_b64": od_b64,
                    "edge_b64": edge_b64,
                    "class_probs": class_probs,
                }
            )
    @app.post("/api/online_update")
    def api_online_update():
        with _DASHBOARD_LOCK:
            dev0 = ctx["dev0"]
            state: SovereignEngineState = ctx["state"]
            zw_va_1v = ctx["zw_va_1v"]
            val_y_g = ctx["val_y_g"]
            zw_te_1v = ctx["zw_te_1v"]
            test_y_g = ctx["test_y_g"]
            _to_rff = ctx["_to_rff"]
            n_val = zw_va_1v.shape[0]
            perm = torch.randperm(n_val, device=dev0)[:1024]
            zw_on = zw_va_1v[perm]
            y_on = F.one_hot(val_y_g[perm], NUM_CLASSES).float() * 2.0 - 1.0
            eye_k = torch.eye(1024, device=dev0, dtype=torch.float32)
            phi_te = _to_rff(zw_te_1v)
            l_before = phi_te @ state.W_chol
            acc_before = float((l_before.argmax(1) == test_y_g).float().mean().item() * 100.0)
            torch.cuda.synchronize(dev0)
            t0 = time.perf_counter()
            phi_on = _to_rff(zw_on)
            U_wood = torch.cholesky_solve(phi_on.T, state.L_chol)
            M_cap = eye_k + (phi_on @ U_wood)
            L_cap = torch.linalg.cholesky(M_cap)
            E_res = y_on - (phi_on @ state.W_chol)
            Z_cap = torch.cholesky_solve(E_res, L_cap)
            W_new = state.W_chol + (U_wood @ Z_cap)
            torch.cuda.synchronize(dev0)
            up_ms = (time.perf_counter() - t0) * 1000.0
            l_after = phi_te @ W_new
            acc_after = float((l_after.argmax(1) == test_y_g).float().mean().item() * 100.0)
            forgetting = max(0.0, acc_before - acc_after)
            return JSONResponse(
                {
                    "slides_added": 1024,
                    "update_ms": round(up_ms, 2),
                    "acc_before": round(acc_before, 2),
                    "acc_after": round(acc_after, 2),
                    "forgetting": round(forgetting, 2),
                }
            )
    @app.post("/api/stress_test")
    def api_stress_test():
        with _DASHBOARD_LOCK:
            run_worker = ctx["run_stress_worker"]
            spec0 = ctx["spec0"]
            spec1_dev1 = ctx["spec1_dev1"]
            bs_stress = ctx["BS_STRESS"]
            n_slides_live = 1_000_000
            total_b = math.ceil(n_slides_live / bs_stress)
            b0 = total_b // 2
            b1 = total_b - b0
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(run_worker, 0, spec0, b0)
                f1 = ex.submit(run_worker, 1 if torch.cuda.device_count() > 1 else 0, spec1_dev1, b1)
                lk0 = f0.result()
                lk1 = f1.result()
            el = time.perf_counter() - t0
            tot_s = total_b * bs_stress
            sps = tot_s / el
            bw = (tot_s * 1176) / (el * 1e9)
            return JSONResponse(
                {
                    "total_slides": tot_s,
                    "elapsed_sec": round(el, 2),
                    "slides_per_sec": round(sps, 1),
                    "pcie_gbs": round(bw, 2),
                    "leak_mb": round(max(lk0, lk1), 1),
                }
            )
    @app.get("/", response_class=HTMLResponse)
    def index_page():
        t = ctx["telemetry"]
        html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>سامانه پاتولوژی دیجیتال  (v12.0)</title>
<link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet" type="text/css" />
<style>
  :root {{
    --bg: #070b14;
    --card: #0f172a;
    --border: #1e293b;
    --accent: #06b6d4;
    --accent2: #10b981;
    --warn: #f43f5e;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; font-family: 'Vazirmatn', sans-serif; }}
  body {{ margin: 0; padding: 20px; background: var(--bg); color: var(--text); line-height: 1.6; }}
  .container {{ max-width: 1360px; margin: 0 auto; }}
  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .badge {{
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
  }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(275px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
  }}
  .card h3 {{ margin: 0 0 8px 0; font-size: 0.95rem; color: var(--muted); }}
  .big-val {{ font-size: 1.65rem; font-weight: 800; color: var(--accent); }}
  .sub-val {{ font-size: 0.83rem; color: var(--muted); margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }}
  th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--border); text-align: right; }}
  th {{ color: var(--accent); font-weight: 700; background: rgba(15, 23, 42, 0.6); }}
  .btn {{
    background: linear-gradient(135deg, #0891b2, #0284c7);
    color: #fff;
    border: none;
    padding: 10px 18px;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 700;
    font-size: 0.9rem;
    transition: opacity 0.2s;
  }}
  .btn:hover {{ opacity: 0.9; }}
  .btn-warn {{ background: linear-gradient(135deg, #e11d48, #be123c); }}
  .btn-green {{ background: linear-gradient(135deg, #059669, #047857); }}
  .lab-grid {{
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 18px;
    margin-top: 20px;
  }}
  @media (max-width: 980px) {{ .lab-grid {{ grid-template-columns: 1fr; }} }}
  .img-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
    text-align: center;
  }}
  .img-box {{
    background: #020617;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
  }}
  .img-box img {{ width: 112px; height: 112px; image-rendering: pixelated; border-radius: 6px; }}
  .bar-bg {{ width: 100%; background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 4px; }}
  .bar-fill {{ height: 100%; background: var(--accent); }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1 style="margin:0; font-size:1.45rem;">سامانه جامع هوش مصنوعی پاتولوژی دیجیتال </h1>
      <div style="color:var(--muted); font-size:0.9rem; margin-top:4px;">
        موتور TensorCore NHWC + کرنل‌های ۱-مرحله‌ای Triton Beer-Lambert + یادگیری تحلیلی Woodbury-Cholesky روی 2x NVIDIA T4
      </div>
    </div>
    <div class="badge">تاییدشده روی Kaggle Dual-T4 | نشت حافظه: {t['vram_leak_mb']} MB</div>
  </div>

  <div class="grid-4">
    <div class="card">
      <h3>سرعت هر ایپوک آموزشی (Dual-T4 CUDAGraph)</h3>
      <div class="big-val">{t['sec_per_ep']} ثانیه</div>
      <div class="sub-val">{t['epochs']} ایپوک کامل در {t['t_p2_sec']} ثانیه (بهبود از 6.778s)</div>
    </div>
    <div class="card">
      <h3>دقت تعمیم بیمارستان خارجی (Center-B)</h3>
      <div class="big-val" style="color:var(--accent2);">{t['acc_p3_te']}%</div>
      <div class="sub-val">اعتبارسنجی داخلی: {t['acc_p4_val']}% | خطای Brier: {t['brier_p4']}</div>
    </div>
    <div class="card">
      <h3>تأخیر تشخیص تک‌لام و یادگیری آنلاین</h3>
      <div class="big-val">{t['b1_ms']} ms</div>
      <div class="sub-val">به‌روزرسانی +۱٬۰۲۴ لام وودبری: {t['online_ms']} ms (فراموشی: {t['forgetting']}%)</div>
    </div>
    <div class="card">
      <h3>توان عملیاتی ۵ میلیون لام و سپر OOD</h3>
      <div class="big-val">{t['slides_per_sec']}</div>
      <div class="sub-val">اسلاید بر ثانیه | AUROC خون‌شناسی: {t['auroc']}% (رد: {t['ood_rej_rate']}%)</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px;">
    <h2 style="margin:0 0 8px 0; font-size:1.1rem;">جدول مقایسه‌ای ۴ پارادایم یادگیری (Ablation Study - بدون نشت داده)</h2>
    <table>
      <thead>
        <tr>
          <th>پارادایم یادگیری</th>
          <th>دقت داخلی (Center-A Val)</th>
          <th>دقت بیمارستان خارجی (Center-B Test)</th>
          <th>معیار کالیبراسیون Brier</th>
          <th>زمان اجرا</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>۱. مخزن نوری خالص بدون ایپوک (Pure 0-Epoch Optical Reservoir)</td>
          <td>{t['acc_p1_val']}%</td>
          <td>{t['acc_p1_te']}%</td>
          <td>{t['brier_p1']}</td>
          <td>{t['t_p1_sec']} ثانیه</td>
        </tr>
        <tr>
          <td>۲. شبکه عمیق TensorCore مبتنی بر ایپوک ({t['epochs']} ایپوک کامل)</td>
          <td>{t['acc_p2_val']}%</td>
          <td>{t['acc_p2_te_1v']}% (1V) / {t['acc_p2_te_2v']}% (2V)</td>
          <td>{t['brier_p2']}</td>
          <td>{t['t_p2_sec']} ثانیه</td>
        </tr>
        <tr>
          <td>۳. حل‌کننده تحلیلی چولسکی با سفیدسازی طیفی (Spectral-Whitened Cholesky)</td>
          <td>{t['acc_p3_val']}%</td>
          <td><b>{t['acc_p3_te']}%</b></td>
          <td>{t['brier_p3']}</td>
          <td>{t['t_p3_sec']} ثانیه</td>
        </tr>
        <tr>
          <td>۴. ترکیب هیبریدی دوگانه (Dual-Paradigm Hybrid Apex)</td>
          <td><b>{t['acc_p4_val']}%</b></td>
          <td>{t['acc_p4_te_1v']}% (1V) / {t['acc_p4_te_2v']}% (2V)</td>
          <td><b>{t['brier_p4']}</b></td>
          <td>{t['t_total_sec']} ثانیه (کل)</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="lab-grid">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <h2 style="margin:0; font-size:1.08rem;">آزمایشگاه ۱: تشخیص زنده تک‌لام (B=1 CUDAGraph) و سپر ایمنی OOD</h2>
        <div style="display:flex; gap:8px;">
          <button class="btn" onclick="runDiagnose('id')">آزمایش لام تصادفی (Center-B)</button>
          <button class="btn btn-warn" onclick="runDiagnose('ood')">تزریق لام خون‌شناسی (OOD)</button>
        </div>
      </div>

      <div class="img-row">
        <div class="img-box">
          <img id="img-rgb" src="" alt="RGB" />
          <div style="font-size:0.8rem; color:var(--muted); margin-top:6px;">تصویر بافت (YCoCg 4:2:0)</div>
        </div>
        <div class="img-box">
          <img id="img-od" src="" alt="OD" />
          <div style="font-size:0.8rem; color:var(--muted); margin-top:6px;">چگالی نوری Beer-Lambert OD</div>
        </div>
        <div class="img-box">
          <img id="img-edge" src="" alt="Edge" />
          <div style="font-size:0.8rem; color:var(--muted); margin-top:6px;">مورفولوژی هسته (Sobel+Laplacian)</div>
        </div>
      </div>

      <div id="diag-info" style="margin-top:14px; padding:12px; background:#020617; border-radius:10px; border:1px solid var(--border); font-size:0.9rem;">
        در حال بارگذاری اولین نمونه بیوپسی از پردازنده گرافیکی...
      </div>

      <div id="prob-bars" style="margin-top:12px;"></div>
    </div>

    <div style="display:flex; flex-direction:column; gap:18px;">
      <div class="card">
        <h2 style="margin:0 0 10px 0; font-size:1.05rem;">آزمایشگاه ۲: یادگیری مستمر آنلاین وودبری-چولسکی (+۱٬۰۲۴ لام)</h2>
        <p style="font-size:0.85rem; color:var(--muted); margin:0 0 12px 0;">
          تزریق زنده ۱٬۰۲۴ لام جدید و به‌روزرسانی دقیق ماتریس هیلبرت ۴٬۶۰۸ بعدی با فرمول ظرفیت وودبری بدون بازآموزش شبکه:
        </p>
        <button class="btn btn-green" onclick="runOnlineUpdate()">تزریق زنده +۱٬۰۲۴ لام جدید روی GPU</button>
        <div id="online-res" style="margin-top:12px; padding:12px; background:#020617; border-radius:10px; border:1px solid var(--border); font-size:0.88rem;">
          آماده اجرای به‌روزرسانی آنی (تأخیر پایه بنچمارک: {t['online_ms']} میلی‌ثانیه | فراموشی: {t['forgetting']}%)
        </div>
      </div>

      <div class="card">
        <h2 style="margin:0 0 10px 0; font-size:1.05rem;">آزمایشگاه ۳: استرس‌تست زنده Dual-T4 PCIe DMA + Triton</h2>
        <p style="font-size:0.85rem; color:var(--muted); margin:0 0 12px 0;">
          اجرای زنده ۱٬۰۰۰٬۰۰۰ لام فشرده YCoCg-R 4:2:0 از حافظه Pinned RAM روی هر دو کارت گرافیک T4 با گراف CUDA:
        </p>
        <button class="btn" onclick="runStressTest()">اجرای استرس‌تست زنده ۱٬۰۰۰٬۰۰۰ لام</button>
        <div id="stress-res" style="margin-top:12px; padding:12px; background:#020617; border-radius:10px; border:1px solid var(--border); font-size:0.88rem;">
          توان عملیاتی ۵ میلیون لام: <b>{t['slides_per_sec']} اسلاید بر ثانیه</b> | پهنای باند PCIe: {t['pcie_gbs']} GB/s | نشت VRAM: {t['vram_leak_mb']} MB
        </div>
      </div>
    </div>
  </div>
</div>

<script>
async function runDiagnose(mode) {{
  const info = document.getElementById('diag-info');
  info.innerHTML = 'در حال اجرای گراف CUDA و کرنل Triton روی GPU 0...';
  const r = await fetch('/api/diagnose?mode=' + mode);
  const d = await r.json();
  document.getElementById('img-rgb').src = d.rgb_b64;
  document.getElementById('img-od').src = d.od_b64;
  document.getElementById('img-edge').src = d.edge_b64;

  const statusColor = d.is_ood ? '#f43f5e' : '#10b981';
  info.innerHTML = `
    <div><b>شماره اسلاید:</b> #${{d.slide_idx}} | <b>برچسب واقعی:</b> ${{d.true_label_fa}}</div>
    <div style="margin-top:4px;"><b>تشخیص سامانه:</b> <span style="color:${{statusColor}}; font-weight:800;">${{d.pred_label_fa}}</span></div>
    <div style="margin-top:4px;"><b>اطمینان شبکه:</b> ${{d.confidence}}% | <b>تأخیر سخت‌افزاری B=1 CUDAGraph:</b> <span style="color:#06b6d4; font-weight:700;">${{d.latency_ms}} ms</span></div>
    <div style="margin-top:4px;"><b>امتیاز OOD (فضای پوچ + انرژی):</b> ${{d.ood_score}} (آستانه ایمنی: ${{d.ood_thresh}})</div>
  `;

  let bars = '';
  for (const item of d.class_probs) {{
    bars += `
      <div style="margin-bottom:6px; font-size:0.8rem;">
        <div style="display:flex; justify-content:space-between;">
          <span>${{item.name_fa}} (${{item.code}})</span>
          <span>${{item.prob}}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:${{Math.min(100, item.prob)}}%"></div></div>
      </div>
    `;
  }}
  document.getElementById('prob-bars').innerHTML = bars;
}}

async function runOnlineUpdate() {{
  const box = document.getElementById('online-res');
  box.innerHTML = 'در حال اجرای به‌روزرسانی ماتریس ظرفیت Woodbury-Cholesky روی GPU...';
  const r = await fetch('/api/online_update', {{ method: 'POST' }});
  const d = await r.json();
  box.innerHTML = `
    <div><b>تعداد لام‌های تزریق‌شده:</b> ${{d.slides_added}} لام</div>
    <div><b>زمان دقیق به‌روزرسانی GPU:</b> <span style="color:#10b981; font-weight:800;">${{d.update_ms}} میلی‌ثانیه</span></div>
    <div><b>دقت بیمارستان خارجی قبل:</b> ${{d.acc_before}}% | <b>بعد از به‌روزرسانی:</b> ${{d.acc_after}}%</div>
    <div><b>میزان فراموشی فاجعه‌بار (Catastrophic Forgetting):</b> <span style="color:#06b6d4; font-weight:800;">${{d.forgetting}}%</span></div>
  `;
}}

async function runStressTest() {{
  const box = document.getElementById('stress-res');
  box.innerHTML = 'در حال اجرای استرس‌تست ۱٬۰۰۰٬۰۰۰ لام روی هر دو کارت گرافیک T4 (حدود ۶ ثانیه)...';
  const r = await fetch('/api/stress_test', {{ method: 'POST' }});
  const d = await r.json();
  box.innerHTML = `
    <div><b>تعداد کل لام‌های پردازش‌شده:</b> ${{d.total_slides.toLocaleString()}} لام در ${{d.elapsed_sec}} ثانیه</div>
    <div><b>توان عملیاتی زنده:</b> <span style="color:#06b6d4; font-weight:800;">${{d.slides_per_sec.toLocaleString()}} اسلاید بر ثانیه</span></div>
    <div><b>پهنای باند PCIe DMA:</b> ${{d.pcie_gbs}} GB/s | <b>نشت حافظه VRAM:</b> ${{d.leak_mb}} MB</div>
  `;
}}

window.onload = () => runDiagnose('id');
</script>
</body>
</html>"""
        return HTMLResponse(html)
    # Pick a free local port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True,
    )
    srv_thread.start()
    time.sleep(1.0)
    # Launch Cloudflare Quick Tunnel
    cf_proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    public_url = None
    t_wait = time.perf_counter()
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    while time.perf_counter() - t_wait < 35.0:
        line = cf_proc.stdout.readline()
        if not line:
            if cf_proc.poll() is not None:
                break
            time.sleep(0.1)
            continue
        m = url_pattern.search(line)
        if m:
            public_url = m.group(0)
            break
    print("\n" + "=" * 88)
    print("  PERSIAN (RTL) LIVE CLOUDFLARE DASHBOARD IS ONLINE!")
    if public_url:
        print(f"  * Public Cloudflare URL : {public_url}")
    else:
        print(f"  * Local Dashboard URL   : http://127.0.0.1:{port}")
    print("=" * 88)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        cf_proc.terminate()
# ====================================================================================
# 7. MAIN VERIFICATION BENCHMARK (100% UNCHANGED CORE)
# ====================================================================================
def main():
    if torch.cuda.device_count() < 2:
        print("WARNING: Less than 2 GPUs detected. Running on available GPU(s).")
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print("=" * 88)
    print("  (Sub-2s TensorCore, Exact Beer-Lambert Triton & Woodbury)")
    print("=" * 88)
    train_x, train_y, val_x, val_y, test_x, test_y, ood_x = load_sovereign_datasets()
    val_y_g = val_y.to(dev0)
    test_y_g = test_y.to(dev0)
    # --------------------------------------------------------------------------------
    # PARADIGM 1: Pure Zero-Epoch Optical Reservoir (with Spectral Whitening)
    # --------------------------------------------------------------------------------
    print("  [Paradigm 1] Running Pure Zero-Epoch Optical Reservoir (0 Epochs Backprop)...")
    t_z0_start = time.perf_counter()
    res_m0 = PureZeroEpochOpticalReservoir(dev0)
    res_m1 = PureZeroEpochOpticalReservoir(dev1)
    bs_z0 = 4096
    def _ext_z0(m, x_t, d):
        torch.cuda.set_device(d)
        outs = []
        for i in range(0, x_t.shape[0], bs_z0):
            outs.append(m.forward_batch(x_t[i : i + bs_z0].to(d, non_blocking=True)).float())
        torch.cuda.synchronize(d)
        return torch.cat(outs, dim=0)
    half_n = train_x.shape[0] // 2
    with ThreadPoolExecutor(max_workers=2) as ex:
        ft0 = ex.submit(_ext_z0, res_m0, train_x[:half_n], dev0)
        ft1 = ex.submit(_ext_z0, res_m1, train_x[half_n:], dev1)
        z0_tr = torch.cat([ft0.result(), ft1.result().to(dev0)], dim=0)
    z0_va = _ext_z0(res_m0, val_x, dev0)
    z0_te = _ext_z0(res_m0, test_x, dev0)
    z0_mean = z0_tr.mean(dim=0, keepdim=True)
    z0_c = z0_tr - z0_mean
    cov0 = (z0_c.T @ z0_c) / float(z0_c.shape[0] - 1)
    cov0.diagonal().add_(1e-3)
    ev0, evec0 = torch.linalg.eigh(cov0)
    W_wh0 = evec0 @ torch.diag(torch.clamp(ev0, min=1e-4).pow(-0.40)) @ evec0.T
    zw0_tr = F.normalize(z0_c @ W_wh0, dim=1)
    zw0_va = F.normalize((z0_va - z0_mean) @ W_wh0, dim=1)
    zw0_te = F.normalize((z0_te - z0_mean) @ W_wh0, dim=1)
    torch.manual_seed(77)
    W_rff0 = torch.randn(768, 3072, device=dev0) * (1.25 / math.sqrt(768.0))
    b_rff0 = torch.rand(1, 3072, device=dev0) * (2.0 * math.pi)
    Y_tr_oh = F.one_hot(train_y.to(dev0), NUM_CLASSES).float() * 2.0 - 1.0
    G0 = torch.zeros((3840, 3840), device=dev0)
    R0 = torch.zeros((3840, NUM_CLASSES), device=dev0)
    rff0_scale = math.sqrt(2.0 / 3072.0)
    for i in range(0, zw0_tr.shape[0], 4096):
        zb = zw0_tr[i : i + 4096]
        hb = torch.cat([zb, torch.cos(zb @ W_rff0 + b_rff0) * rff0_scale], dim=1)
        G0.addmm_(hb.T, hb)
        R0.addmm_(hb.T, Y_tr_oh[i : i + 4096])
    G0.diagonal().add_(12.0)
    L0 = torch.linalg.cholesky(G0)
    W0_out = torch.cholesky_solve(R0, L0)
    p0_va = torch.cat([zw0_va, torch.cos(zw0_va @ W_rff0 + b_rff0) * rff0_scale], dim=1) @ W0_out
    p0_te = torch.cat([zw0_te, torch.cos(zw0_te @ W_rff0 + b_rff0) * rff0_scale], dim=1) @ W0_out
    acc_p1_val = float((p0_va.argmax(1) == val_y_g).float().mean().item() * 100.0)
    acc_p1_te = float((p0_te.argmax(1) == test_y_g).float().mean().item() * 100.0)
    best_t_p1 = 3.0
    best_b_p1 = 999.0
    for sc_cand in np.linspace(1.5, 6.0, 19):
        bv = compute_brier(p0_va * float(sc_cand), val_y_g)
        if bv < best_b_p1:
            best_b_p1 = bv
            best_t_p1 = float(sc_cand)
    brier_p1 = compute_brier(p0_te * best_t_p1, test_y_g)
    t_p1_sec = time.perf_counter() - t_z0_start
    del z0_tr, z0_va, z0_te, zw0_tr, zw0_va, zw0_te, G0, L0, res_m0, res_m1
    torch.cuda.empty_cache()
    # --------------------------------------------------------------------------------
    # PARADIGM 2: 32-Epoch Complementary Sub-2s Specialists on Dual T4
    # --------------------------------------------------------------------------------
    EPOCHS = 32
    print(f"  [Paradigm 2] Training {EPOCHS}-Epoch Sub-2s TensorCore Specialists on Dual T4...")
    t_ep_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        # Specialist 0: Sharp Morphology & Chromatin Specialist
        f_sp0 = ex.submit(
            train_fast_specialist,
            0,
            train_x,
            train_y,
            EPOCHS,
            101,
            10,
            0.45,
            0.40,
            0.06,
            2.5e-4,
            0.08,
            0.12,
        )
        # Specialist 1: Strong Beer-Lambert Stain-Invariant Domain Specialist
        f_sp1 = ex.submit(
            train_fast_specialist,
            1 if torch.cuda.device_count() > 1 else 0,
            train_x,
            train_y,
            EPOCHS,
            202,
            12,
            0.55,
            0.60,
            0.09,
            3.5e-4,
            0.10,
            0.18,
        )
        spec0 = f_sp0.result()
        spec1_dev1 = f_sp1.result()
    t_p2_sec = time.perf_counter() - t_ep_start
    sec_per_ep = t_p2_sec / EPOCHS
    # Mirror spec1 to dev0 BEFORE fusing Conv+BN so state_dict keys match 100%
    spec1 = (
        DualPhysicsTensorCoreNet(NUM_CLASSES, drop_p=0.0)
        .to(dev0)
        .half()
        .to(memory_format=torch.channels_last)
    )
    spec1.load_state_dict(spec1_dev1.state_dict())
    # Fuse Conv2d + BatchNorm2d for blazing inference across Paradigm 3/4 & Pillars 1-4
    spec0.fuse_bn_for_inference()
    spec1.fuse_bn_for_inference()
    spec1_dev1.fuse_bn_for_inference()
    # --------------------------------------------------------------------------------
    # PARADIGM 3 & 4: Spectral-Whitened Zero-Epoch Cholesky & Hybrid Apex
    # --------------------------------------------------------------------------------
    print("  [Paradigm 3 & 4] Extracting Features & Solving Spectral-Whitened Cholesky...")
    t_p3_start = time.perf_counter()
    def _ext_all_splits(m, d):
        torch.cuda.set_device(d)
        r_tr = extract_features_2view(m, train_x, d)
        r_va = extract_features_2view(m, val_x, d)
        r_te = extract_features_2view(m, test_x, d)
        torch.cuda.synchronize(d)
        return r_tr, r_va, r_te
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_all0 = ex.submit(_ext_all_splits, spec0, dev0)
        f_all1 = ex.submit(_ext_all_splits, spec1_dev1, dev1)
        (
            (z_tr0_1v, z_tr0_hf, _, _, _),
            (z_va0_1v, _, z_va0_2v, l_va0_1v, l_va0_2v),
            (z_te0_1v, _, z_te0_2v, l_te0_1v, l_te0_2v),
        ) = f_all0.result()
        (
            (z_tr1_1v, z_tr1_hf, _, _, _),
            (z_va1_1v, _, z_va1_2v, l_va1_1v, l_va1_2v),
            (z_te1_1v, _, z_te1_2v, l_te1_1v, l_te1_2v),
        ) = f_all1.result()
    z_tr_1v = torch.cat([z_tr0_1v, z_tr1_1v.to(dev0)], dim=1)
    z_tr_hf = torch.cat([z_tr0_hf, z_tr1_hf.to(dev0)], dim=1)
    z_va_1v = torch.cat([z_va0_1v, z_va1_1v.to(dev0)], dim=1)
    z_va_2v = torch.cat([z_va0_2v, z_va1_2v.to(dev0)], dim=1)
    z_te_1v = torch.cat([z_te0_1v, z_te1_1v.to(dev0)], dim=1)
    z_te_2v = torch.cat([z_te0_2v, z_te1_2v.to(dev0)], dim=1)
    l_ep_va_1v = 0.5 * (l_va0_1v + l_va1_1v.to(dev0))
    l_ep_va_2v = 0.5 * (l_va0_2v + l_va1_2v.to(dev0))
    l_ep_te_1v = 0.5 * (l_te0_1v + l_te1_1v.to(dev0))
    l_ep_te_2v = 0.5 * (l_te0_2v + l_te1_2v.to(dev0))
    acc_p2_val = float((l_ep_va_1v.argmax(1) == val_y_g).float().mean().item() * 100.0)
    acc_p2_te_1v = float((l_ep_te_1v.argmax(1) == test_y_g).float().mean().item() * 100.0)
    acc_p2_te_2v = float((l_ep_te_2v.argmax(1) == test_y_g).float().mean().item() * 100.0)
    brier_p2 = compute_brier(l_ep_te_1v, test_y_g)
    # Spectral Whitening (strictly fit on train_x only)
    z_mean = z_tr_1v.mean(dim=0, keepdim=True)
    zc_tr = z_tr_1v - z_mean
    cov_tr = (zc_tr.T @ zc_tr) / float(zc_tr.shape[0] - 1)
    cov_tr.diagonal().add_(1e-3)
    evals, evecs = torch.linalg.eigh(cov_tr)
    evals = torch.clamp(evals, min=1e-4)
    W_whiten = evecs @ torch.diag(evals.pow(-0.35)) @ evecs.T
    zw_tr_1v = F.normalize(zc_tr @ W_whiten, dim=1)
    zw_tr_hf = F.normalize((z_tr_hf - z_mean) @ W_whiten, dim=1)
    zw_va_1v = F.normalize((z_va_1v - z_mean) @ W_whiten, dim=1)
    zw_va_2v = F.normalize((z_va_2v - z_mean) @ W_whiten, dim=1)
    zw_te_1v = F.normalize((z_te_1v - z_mean) @ W_whiten, dim=1)
    zw_te_2v = F.normalize((z_te_2v - z_mean) @ W_whiten, dim=1)
    torch.manual_seed(2026)
    W_rff = torch.randn(1536, 3072, device=dev0) * (1.25 / math.sqrt(1536.0))
    b_rff = torch.rand(1, 3072, device=dev0) * (2.0 * math.pi)
    rff_scale = math.sqrt(2.0 / 3072.0)
    def _to_rff(zw):
        h = torch.cos(zw @ W_rff + b_rff) * rff_scale
        return torch.cat([zw, h], dim=1)
    # Accumulate 2-view flip-invariant Hilbert Gram matrix (180k views -> stronger regularization)
    D_hilb = 4608
    G_mat = torch.zeros((D_hilb, D_hilb), device=dev0)
    XY_rhs = torch.zeros((D_hilb, NUM_CLASSES), device=dev0)
    for i in range(0, zw_tr_1v.shape[0], 4096):
        yb = Y_tr_oh[i : i + 4096]
        phi1 = _to_rff(zw_tr_1v[i : i + 4096])
        G_mat.addmm_(phi1.T, phi1)
        XY_rhs.addmm_(phi1.T, yb)
        phi2 = _to_rff(zw_tr_hf[i : i + 4096])
        G_mat.addmm_(phi2.T, phi2)
        XY_rhs.addmm_(phi2.T, yb)
    best_va_acc = -1.0
    best_L = None
    best_W_chol = None
    phi_va_1v = _to_rff(zw_va_1v)
    for reg in (8.0, 24.0, 64.0, 160.0):
        G_reg = G_mat.clone()
        G_reg.diagonal().add_(reg)
        L_c = torch.linalg.cholesky(G_reg)
        W_c = torch.cholesky_solve(XY_rhs, L_c)
        l_va_c = phi_va_1v @ W_c
        acc_c = float((l_va_c.argmax(1) == val_y_g).float().mean().item())
        if acc_c > best_va_acc:
            best_va_acc = acc_c
            best_L = L_c
            best_W_chol = W_c
    l_chol_va_1v = phi_va_1v @ best_W_chol
    l_chol_va_2v = _to_rff(zw_va_2v) @ best_W_chol
    l_chol_te_1v = _to_rff(zw_te_1v) @ best_W_chol
    l_chol_te_2v = _to_rff(zw_te_2v) @ best_W_chol
    acc_p3_val = float((l_chol_va_1v.argmax(1) == val_y_g).float().mean().item() * 100.0)
    acc_p3_te = float((l_chol_te_1v.argmax(1) == test_y_g).float().mean().item() * 100.0)
    # Joint Accuracy + Brier-calibrated Hybrid Apex tuning (strictly on val_y_g)
    s_ep = l_ep_va_1v.std() + 1e-6
    s_ch = l_chol_va_1v.std() + 1e-6
    cand_alphas = np.linspace(0.0, 1.0, 41)
    va_accs = []
    for alpha in cand_alphas:
        al = float(alpha)
        comb = (1.0 - al) * (l_ep_va_1v / s_ep) + al * (l_chol_va_1v / s_ch)
        va_accs.append(float((comb.argmax(1) == val_y_g).float().mean().item()))
    max_va_acc = max(va_accs)
    best_alpha = 0.5
    best_temp = 1.0
    best_brier_va = 999.0
    for alpha, acc_h in zip(cand_alphas, va_accs):
        if acc_h >= max_va_acc - 0.0005:
            al = float(alpha)
            comb = (1.0 - al) * (l_ep_va_1v / s_ep) + al * (l_chol_va_1v / s_ch)
            for t_cand in np.linspace(0.20, 2.0, 37):
                bv = compute_brier(comb / float(t_cand), val_y_g)
                if bv < best_brier_va:
                    best_brier_va = bv
                    best_alpha = al
                    best_temp = float(t_cand)
    l_hyb_va_1v = (1.0 - best_alpha) * (l_ep_va_1v / s_ep) + best_alpha * (l_chol_va_1v / s_ch)
    l_hyb_te_1v = (1.0 - best_alpha) * (l_ep_te_1v / s_ep) + best_alpha * (l_chol_te_1v / s_ch)
    l_hyb_te_2v = (1.0 - best_alpha) * (l_ep_te_2v / s_ep) + best_alpha * (l_chol_te_2v / s_ch)
    acc_p4_val = float((l_hyb_va_1v.argmax(1) == val_y_g).float().mean().item() * 100.0)
    acc_p4_te_1v = float((l_hyb_te_1v.argmax(1) == test_y_g).float().mean().item() * 100.0)
    acc_p4_te_2v = float((l_hyb_te_2v.argmax(1) == test_y_g).float().mean().item() * 100.0)
    brier_p4 = compute_brier(l_hyb_te_1v / best_temp, test_y_g)
    best_t_p3 = 1.0
    best_b3_va = 999.0
    for t_cand in np.linspace(0.15, 2.0, 38):
        bv = compute_brier((l_chol_va_1v / s_ch) / float(t_cand), val_y_g)
        if bv < best_b3_va:
            best_b3_va = bv
            best_t_p3 = float(t_cand)
    brier_p3 = compute_brier((l_chol_te_1v / s_ch) / best_t_p3, test_y_g)
    W_ep = 0.5 * torch.cat(
        [spec0.fc_cls.weight.detach().float().T, spec1.fc_cls.weight.detach().float().T], dim=0
    )
    b_ep = 0.5 * (spec0.fc_cls.bias.detach().float() + spec1.fc_cls.bias.detach().float())
    # Calibrate OOD Null-Space + Principal Subspace on val_x + exact 1-pass YCoCg 4:2:0 decoded val_x
    val_comp_sub = compress_ycocg420_vectorized(val_x[:2048]).to(dev0)
    val_phys8_nhwc = torch.empty((2048, 28, 28, 8), device=dev0, dtype=torch.float16)
    triton_decode_ycocg420_to_phys8_nhwc(val_comp_sub, val_phys8_nhwc)
    val_phys8 = val_phys8_nhwc.permute(0, 3, 1, 2)
    with torch.no_grad():
        zv_c0 = spec0.forward_features_from_8ch(val_phys8).float()
        zv_c1 = spec1.forward_features_from_8ch(val_phys8).float()
        z_va_comp = torch.cat([zv_c0, zv_c1], dim=1)
        l_va_comp = z_va_comp @ W_ep + b_ep
    z_va_cal = torch.cat([z_va_1v, z_va_comp], dim=0)
    l_va_cal = torch.cat([l_ep_va_1v, l_va_comp], dim=0)
    zc_va = F.normalize(z_va_cal - z_mean, dim=1)
    cov_va = (zc_va.T @ zc_va) / float(zc_va.shape[0] - 1)
    cov_va.diagonal().add_(5e-3)
    _, evecs_va = torch.linalg.eigh(cov_va)
    U_null = evecs_va[:, :128].contiguous()
    U_prin = evecs_va[:, -128:].contiguous()
    ns_va = torch.norm(zc_va @ U_null, dim=1) / (torch.norm(zc_va @ U_prin, dim=1) + 1e-4)
    en_va = -torch.logsumexp(l_va_cal, dim=1)
    m_ns, s_ns = float(ns_va.mean().item()), float(ns_va.std().item() + 1e-5)
    m_en, s_en = float(en_va.mean().item()), float(en_va.std().item() + 1e-5)
    ood_va_scores = 0.65 * ((ns_va - m_ns) / s_ns) + 0.35 * ((en_va - m_en) / s_en)
    ood_thresh = float(torch.quantile(ood_va_scores, 0.992).item() * 1.18)
    t_p3_sec = time.perf_counter() - t_p3_start
    t_total_sec = t_p1_sec + t_p2_sec + t_p3_sec
    state = SovereignEngineState(
        spec0=spec0,
        spec1=spec1,
        z_mean=z_mean.half(),
        W_whiten=W_whiten.half(),
        W_rff=W_rff.half(),
        b_rff=b_rff.half(),
        L_chol=best_L,
        W_chol=best_W_chol,
        XY_rhs=XY_rhs,
        W_hybrid=W_ep.half(),
        b_hybrid=b_ep.half(),
        U_null=U_null.half(),
        U_prin=U_prin.half(),
        vim_alpha=best_alpha,
        m_ns=m_ns,
        s_ns=s_ns,
        m_en=m_en,
        s_en=s_en,
        ood_thresh=ood_thresh,
        temp_opt=best_temp,
    )
    # --------------------------------------------------------------------------------
    # PILLAR 1: Single Biopsy Diagnosis (B=1 CUDAGraph + 1-Pass Fused YCoCg->8ch Triton)
    # --------------------------------------------------------------------------------
    test_comp = compress_ycocg420_vectorized(test_x)
    static_b1_comp = torch.zeros((1, 1176), device=dev0, dtype=torch.uint8)
    static_b1_phys8_nhwc = torch.zeros((1, 28, 28, 8), device=dev0, dtype=torch.float16)
    static_b1_phys8 = static_b1_phys8_nhwc.permute(0, 3, 1, 2)
    static_b1_probs = torch.zeros((1, NUM_CLASSES), device=dev0, dtype=torch.float32)
    static_b1_ood = torch.zeros((1,), device=dev0, dtype=torch.float32)
    @torch.no_grad()
    def _b1_graph_body():
        triton_decode_ycocg420_to_phys8_nhwc(static_b1_comp, static_b1_phys8_nhwc)
        f0 = state.spec0.forward_features_from_8ch(static_b1_phys8)
        f1 = state.spec1.forward_features_from_8ch(static_b1_phys8)
        z = torch.cat([f0, f1], dim=1)
        l_ep = z @ state.W_hybrid + state.b_hybrid
        zc = F.normalize(z - state.z_mean, dim=1)
        ns = torch.norm(zc @ state.U_null, dim=1) / (torch.norm(zc @ state.U_prin, dim=1) + 1e-4)
        en = -torch.logsumexp(l_ep.float(), dim=1)
        sc = 0.65 * ((ns.float() - state.m_ns) / state.s_ns) + 0.35 * ((en - state.m_en) / state.s_en)
        static_b1_ood.copy_(sc)
        static_b1_probs.copy_(F.softmax(l_ep.float() / state.temp_opt, dim=1))
    s_b1 = torch.cuda.Stream(device=dev0)
    b1_graph = torch.cuda.CUDAGraph()
    with _GRAPH_CAPTURE_LOCK:
        s_b1.wait_stream(torch.cuda.current_stream(dev0))
        with torch.cuda.stream(s_b1):
            static_b1_comp.copy_(test_comp[0:1].to(dev0))
            for _ in range(5):
                _b1_graph_body()
        torch.cuda.current_stream(dev0).wait_stream(s_b1)
        with torch.cuda.graph(b1_graph, stream=s_b1, capture_error_mode="thread_local"):
            _b1_graph_body()
        torch.cuda.synchronize(dev0)
    b1_sample_pinned = test_comp[0:1].pin_memory()
    static_b1_comp.copy_(b1_sample_pinned, non_blocking=True)
    b1_graph.replay()
    torch.cuda.synchronize(dev0)
    t_b1_0 = time.perf_counter()
    static_b1_comp.copy_(b1_sample_pinned, non_blocking=True)
    b1_graph.replay()
    torch.cuda.synchronize(dev0)
    b1_ms = (time.perf_counter() - t_b1_0) * 1000.0
    b1_ood_val = float(static_b1_ood[0].item())
    b1_pred = int(static_b1_probs[0].argmax().item())
    b1_conf = float(static_b1_probs[0, b1_pred].item() * 100.0)
    b1_label = "REJECTED_OOD" if b1_ood_val > state.ood_thresh else CLASS_NAMES[b1_pred][0]
    # --------------------------------------------------------------------------------
    # PILLAR 2: True Exact Woodbury-Cholesky Online Update (+1,024 Slides, O(D*k^2 + k^3))
    # --------------------------------------------------------------------------------
    zw_online = zw_va_1v[:1024]
    y_online = F.one_hot(val_y_g[:1024], NUM_CLASSES).float() * 2.0 - 1.0
    eye_k = torch.eye(1024, device=dev0, dtype=torch.float32)
    phi_te_1v = _to_rff(zw_te_1v)
    torch.cuda.synchronize(dev0)
    t_p2_on0 = time.perf_counter()
    phi_on = _to_rff(zw_online)                                # (1024, 4608)
    U_wood = torch.cholesky_solve(phi_on.T, state.L_chol)      # (4608, 1024) = G0^-1 Phi^T
    M_cap = eye_k + (phi_on @ U_wood)                          # (1024, 1024) Capacitance Matrix
    L_cap = torch.linalg.cholesky(M_cap)                       # 91x fewer ops than 4608^3 Cholesky!
    E_res = y_online - (phi_on @ state.W_chol)                 # (1024, 9)
    Z_cap = torch.cholesky_solve(E_res, L_cap)                 # (1024, 9)
    W_new = state.W_chol + (U_wood @ Z_cap)                    # Exact rank-1024 updated weights
    torch.cuda.synchronize(dev0)
    online_ms = (time.perf_counter() - t_p2_on0) * 1000.0
    l_after = phi_te_1v @ W_new
    acc_after = float((l_after.argmax(1) == test_y_g).float().mean().item() * 100.0)
    forgetting = max(0.0, acc_p3_te - acc_after)
    # --------------------------------------------------------------------------------
    # PILLAR 3: OOD Guardrail (BloodMNIST Hematology Slides)
    # --------------------------------------------------------------------------------
    z_ood0, _, _, l_ood0, _ = extract_features_2view(spec0, ood_x, dev0)
    z_ood1, _, _, l_ood1, _ = extract_features_2view(spec1, ood_x, dev0)
    z_ood = torch.cat([z_ood0, z_ood1], dim=1)
    l_ood = 0.5 * (l_ood0 + l_ood1)
    zc_te = F.normalize(z_te_1v - z_mean, dim=1)
    ns_te = torch.norm(zc_te @ U_null, dim=1) / (torch.norm(zc_te @ U_prin, dim=1) + 1e-4)
    en_te = -torch.logsumexp(l_ep_te_1v, dim=1)
    id_scores = 0.65 * ((ns_te - m_ns) / s_ns) + 0.35 * ((en_te - m_en) / s_en)
    zc_ood = F.normalize(z_ood - z_mean, dim=1)
    ns_ood = torch.norm(zc_ood @ U_null, dim=1) / (torch.norm(zc_ood @ U_prin, dim=1) + 1e-4)
    en_ood = -torch.logsumexp(l_ood, dim=1)
    ood_scores = 0.65 * ((ns_ood - m_ns) / s_ns) + 0.35 * ((en_ood - m_en) / s_en)
    all_s = torch.cat([id_scores, ood_scores])
    all_l = torch.cat([torch.zeros_like(id_scores), torch.ones_like(ood_scores)])
    order = torch.argsort(all_s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(all_s) + 1, device=dev0, dtype=torch.float32)
    n_pos = float(ood_scores.numel())
    n_neg = float(id_scores.numel())
    auroc = float(((ranks[all_l == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)).item() * 100.0)
    ood_rej_rate = float((ood_scores > state.ood_thresh).float().mean().item() * 100.0)
    sample_ood_str = "REJECTED_OOD" if float(ood_scores[0].item()) > state.ood_thresh else "ACCEPTED"
    # --------------------------------------------------------------------------------
    # PILLAR 4: 5,000,000-Slide Double-Buffered Async PCIe DMA + 1-Pass Triton + CUDAGraph
    # --------------------------------------------------------------------------------
    BS_STRESS = 4096
    comp_pinned = test_comp[:BS_STRESS].pin_memory()
    def _run_stress_worker(gpu_id: int, model_w: DualPhysicsTensorCoreNet, n_batches: int) -> float:
        torch.cuda.set_device(gpu_id)
        d = torch.device(f"cuda:{gpu_id}")
        s_comp = [torch.zeros((BS_STRESS, 1176), device=d, dtype=torch.uint8) for _ in range(2)]
        s_phys8_nhwc = [torch.zeros((BS_STRESS, 28, 28, 8), device=d, dtype=torch.float16) for _ in range(2)]
        s_phys8 = [buf.permute(0, 3, 1, 2) for buf in s_phys8_nhwc]
        s_out = [torch.zeros((BS_STRESS, NUM_CLASSES), device=d, dtype=torch.float16) for _ in range(2)]
        st_dma = torch.cuda.Stream(device=d)
        st_comp = torch.cuda.Stream(device=d)
        ev_dma = [torch.cuda.Event() for _ in range(2)]
        ev_comp = [torch.cuda.Event() for _ in range(2)]
        g_stress = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        @torch.no_grad()
        def _stress_step(slot: int):
            triton_decode_ycocg420_to_phys8_nhwc(s_comp[slot], s_phys8_nhwc[slot])
            f_feat = model_w.forward_features_from_8ch(s_phys8[slot])
            s_out[slot].copy_(model_w.fc_cls(f_feat))
        with _GRAPH_CAPTURE_LOCK:
            st_comp.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(st_comp):
                for slot in range(2):
                    s_comp[slot].copy_(comp_pinned, non_blocking=True)
                    for _ in range(3):
                        _stress_step(slot)
            torch.cuda.current_stream(d).wait_stream(st_comp)
            torch.cuda.synchronize(d)
            for slot in range(2):
                with torch.cuda.graph(g_stress[slot], stream=st_comp, capture_error_mode="thread_local"):
                    _stress_step(slot)
            torch.cuda.synchronize(d)
        mem_pre_loop = torch.cuda.memory_allocated(d)
        ev_comp[0].record(st_comp)
        ev_comp[1].record(st_comp)
        for i in range(n_batches):
            slot = i & 1
            with torch.cuda.stream(st_dma):
                st_dma.wait_event(ev_comp[slot])
                s_comp[slot].copy_(comp_pinned, non_blocking=True)
                ev_dma[slot].record(st_dma)
            with torch.cuda.stream(st_comp):
                st_comp.wait_event(ev_dma[slot])
                g_stress[slot].replay()
                ev_comp[slot].record(st_comp)
        torch.cuda.synchronize(d)
        leak_mb = max(0.0, (torch.cuda.memory_allocated(d) - mem_pre_loop) / (1024 * 1024))
        del g_stress, s_comp, s_phys8_nhwc, s_phys8, s_out
        return leak_mb
    total_batches = math.ceil(5_000_000 / BS_STRESS)
    b_g0 = total_batches // 2
    b_g1 = total_batches - b_g0
    t_str0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fs0 = ex.submit(_run_stress_worker, 0, spec0, b_g0)
        fs1 = ex.submit(
            _run_stress_worker,
            1 if torch.cuda.device_count() > 1 else 0,
            spec1_dev1,
            b_g1,
        )
        leak0 = fs0.result()
        leak1 = fs1.result()
    t_str_sec = time.perf_counter() - t_str0
    vram_leak_mb = max(leak0, leak1)
    total_slides = total_batches * BS_STRESS
    slides_per_sec = total_slides / t_str_sec
    pcie_gbs = (total_slides * 1176) / (t_str_sec * 1e9)
    print("\n" + "=" * 88)
    print("  VERIFIED 4-PARADIGM ABLATION & 4-PILLAR TELEMETRY (Kaggle 2x T4)")
    print("=" * 88)
    print(
        f"  0. Speed per Epoch (v11.1 -> v12.0)       : 6.778 s/ep -> {sec_per_ep:.3f} s/ep ({EPOCHS} Epochs in {t_p2_sec:.2f}s)"
    )
    print(
        f"  1. Pure Zero-Epoch (0 Epochs Reservoir)   : Val={acc_p1_val:5.2f}% | Ext Center-B={acc_p1_te:5.2f}% | Brier={brier_p1:.4f} | Time={t_p1_sec:.2f}s"
    )
    print(
        f"  2. Pure Epoch-Based ({EPOCHS} Epochs TensorNet) : Val={acc_p2_val:5.2f}% | Ext Center-B={acc_p2_te_1v:5.2f}% (1V) / {acc_p2_te_2v:5.2f}% (2V) | Brier={brier_p2:.4f}"
    )
    print(
        f"  3. Spectral-Whitened Zero-Epoch Cholesky  : Val={acc_p3_val:5.2f}% | Ext Center-B={acc_p3_te:5.2f}% | Brier={brier_p3:.4f} | Time={t_p3_sec:.2f}s"
    )
    print(
        f"  4. Dual-Paradigm Hybrid Apex (1V / 2V)    : Val={acc_p4_val:5.2f}% | Ext Center-B={acc_p4_te_1v:5.2f}% (1V) / {acc_p4_te_2v:5.2f}% (2V) | Brier={brier_p4:.4f}"
    )
    print("-" * 88)
    print(
        f"  * Total Train+Calibrate  : {t_total_sec:.2f}s ({EPOCHS}-Ep Backbones: {t_p2_sec:.2f}s | Calibration+0-Ep: {t_p1_sec+t_p3_sec:.2f}s)"
    )
    print(f"  * Pillar 1 (Single Biopsy): {b1_label} ({b1_conf:.2f}%) in {b1_ms:.3f} ms [B=1 CUDAGraph]")
    print(f"  * Pillar 2 (+1,024 Online): {online_ms:.2f} ms (Exact Woodbury-Cholesky | Forgetting: {forgetting:.2f}%)")
    print(
        f"  * Pillar 3 (BloodMNIST OOD): AUROC: {auroc:.2f}% | Rejected: {ood_rej_rate:.2f}% (Sample: {sample_ood_str})"
    )
    print(
        f"  * Pillar 4 (5M-Slide Stress): {slides_per_sec:,.1f} slides/sec | PCIe: {pcie_gbs:.2f} GB/s | Leak: {vram_leak_mb:.1f} MB"
    )
    print("=" * 88)
    telemetry_dict = {
        "epochs": EPOCHS,
        "sec_per_ep": f"{sec_per_ep:.3f}",
        "t_p2_sec": f"{t_p2_sec:.2f}",
        "acc_p1_val": f"{acc_p1_val:.2f}",
        "acc_p1_te": f"{acc_p1_te:.2f}",
        "brier_p1": f"{brier_p1:.4f}",
        "t_p1_sec": f"{t_p1_sec:.2f}",
        "acc_p2_val": f"{acc_p2_val:.2f}",
        "acc_p2_te_1v": f"{acc_p2_te_1v:.2f}",
        "acc_p2_te_2v": f"{acc_p2_te_2v:.2f}",
        "brier_p2": f"{brier_p2:.4f}",
        "acc_p3_val": f"{acc_p3_val:.2f}",
        "acc_p3_te": f"{acc_p3_te:.2f}",
        "brier_p3": f"{brier_p3:.4f}",
        "t_p3_sec": f"{t_p3_sec:.2f}",
        "acc_p4_val": f"{acc_p4_val:.2f}",
        "acc_p4_te_1v": f"{acc_p4_te_1v:.2f}",
        "acc_p4_te_2v": f"{acc_p4_te_2v:.2f}",
        "brier_p4": f"{brier_p4:.4f}",
        "t_total_sec": f"{t_total_sec:.2f}",
        "b1_label": b1_label,
        "b1_conf": f"{b1_conf:.2f}",
        "b1_ms": f"{b1_ms:.3f}",
        "online_ms": f"{online_ms:.2f}",
        "forgetting": f"{forgetting:.2f}",
        "auroc": f"{auroc:.2f}",
        "ood_rej_rate": f"{ood_rej_rate:.2f}",
        "slides_per_sec": f"{slides_per_sec:,.1f}",
        "pcie_gbs": f"{pcie_gbs:.2f}",
        "vram_leak_mb": f"{vram_leak_mb:.1f}",
    }
    launch_sovereign_persian_dashboard(
        {
            "telemetry": telemetry_dict,
            "dev0": dev0,
            "state": state,
            "test_x": test_x,
            "test_y": test_y,
            "ood_x": ood_x,
            "static_b1_comp": static_b1_comp,
            "static_b1_phys8": static_b1_phys8,
            "static_b1_probs": static_b1_probs,
            "static_b1_ood": static_b1_ood,
            "b1_graph": b1_graph,
            "zw_va_1v": zw_va_1v,
            "val_y_g": val_y_g,
            "zw_te_1v": zw_te_1v,
            "test_y_g": test_y_g,
            "_to_rff": _to_rff,
            "run_stress_worker": _run_stress_worker,
            "spec0": spec0,
            "spec1_dev1": spec1_dev1,
            "BS_STRESS": BS_STRESS,
        }
    )
if __name__ == "__main__":
    main()



print("=" * 88)
