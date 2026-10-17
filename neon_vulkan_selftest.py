#!/usr/bin/env python3
"""
neon_vulkan_selftest.py - one-command test for NEON's Vulkan backend.

Run from the repo root (branch VULKAN-EXPERIMENTAL, where the matvec_*.comp
shaders live):

    python3 neon_vulkan_selftest.py

What it does, automatically:
  1. finds a shader compiler (glslangValidator or glslc), a C++ compiler and
     the Vulkan headers/loader, and INSTALLS whatever is missing: first via the
     system package manager (apt/dnf/pacman/brew, using sudo -n if you are not
     root), then, if that is impossible, by downloading a prebuilt glslang from
     Khronos' GitHub release and the Vulkan-Headers sources into the build dir
     (no root needed). Disable with --no-auto-install
  2. compiles EVERY *.comp in the repo to SPIR-V, trying several --target-env
     levels and both compilers, plus the INT8_STORAGE (_int8) and dp4a variants
     where the sources exist. Required vs optional shaders are tracked, and an
     optional-shader failure is reported but never aborts the run
  3. if matmul_*.comp sources are absent (they are, on the current branches)
     it makes a PATCHED COPY of rawllm_vulkan.hpp in the build dir so the
     backend constructor no longer requires them. Your repo is never modified
  4. generates a C++ harness, builds it against the real header, runs it
  5. checks correctness against a CPU double-precision reference for: F32
     matvec (vec4 + scalar paths), x2, multirow 1/2/4, F16, Q4_0, Q8_0,
     stale-data stress (descriptor-dedup path), multi-op batching and matmul
     F32 (only if its shader exists), then benchmarks
  6. prints a table, writes vulkan_selftest_report.json, exit code 0 = all pass

Useful flags:  --full  (bigger shapes)   --no-bench   --shaders-only
               --lavapipe (force Mesa software Vulkan)   --no-auto-install
"""
import argparse, io, json, os, platform, re, shutil, stat, subprocess, sys, tarfile, textwrap, time, urllib.request, zipfile
from pathlib import Path

REQUIRED_SHADERS = ["matvec_f32", "matvec_f32_scalar", "matvec_q4_0", "matvec_q8_0"]
# name -> list of (output_suffix, extra glslang defines)
VARIANTS = {
    "matvec_q4_0": [("_int8", ["INT8_STORAGE=1"])],
    "matvec_q8_0": [("_int8", ["INT8_STORAGE=1"])],
    "matmul_tiled_q4_0": [("_dp4a", ["HAS_INT8_DOT=1"])],
    "matmul_tiled_q8_0": [("_dp4a", ["HAS_INT8_DOT=1"])],
}
TARGET_ENVS = ["vulkan1.1", "vulkan1.2", "vulkan1.3"]

APT_PKGS = "glslang-tools libvulkan-dev mesa-vulkan-drivers vulkan-tools build-essential"
INSTALL_HINTS = {
    "apt": f"sudo apt-get install -y {APT_PKGS}",
    "dnf": "sudo dnf install -y glslang vulkan-loader-devel vulkan-headers mesa-vulkan-drivers vulkan-tools gcc-c++",
    "pacman": "sudo pacman -S --needed glslang vulkan-headers vulkan-icd-loader vulkan-tools vulkan-swrast base-devel",
    "brew": "brew install glslang vulkan-headers vulkan-loader molten-vk",
    "win": "Install the LunarG Vulkan SDK (gives glslc, glslangValidator, headers, loader) and build under MSYS2/MinGW or WSL2.",
}
# what to install per missing component, per package manager
PKGS = {
    "apt":    {"shader": ["glslang-tools"], "vkdev": ["libvulkan-dev"], "cxx": ["build-essential"], "runtime": ["mesa-vulkan-drivers", "vulkan-tools"]},
    "dnf":    {"shader": ["glslang"], "vkdev": ["vulkan-loader-devel", "vulkan-headers"], "cxx": ["gcc-c++"], "runtime": ["mesa-vulkan-drivers", "vulkan-tools"]},
    "pacman": {"shader": ["glslang"], "vkdev": ["vulkan-headers", "vulkan-icd-loader"], "cxx": ["base-devel"], "runtime": ["vulkan-swrast", "vulkan-tools"]},
    "brew":   {"shader": ["glslang"], "vkdev": ["vulkan-headers", "vulkan-loader"], "cxx": [], "runtime": ["molten-vk", "vulkan-tools"]},
}
GLSLANG_KEYS = {"Linux": "linux-x86_64", "Darwin": "macos-universal", "Windows": "windows-x86_64"}
GLSLANG_BASE = "https://github.com/KhronosGroup/glslang/releases"
VKHEADERS_URL = "https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/refs/heads/main"


def sh(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


def which_any(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def pkg_manager():
    if platform.system() == "Windows":
        return "win"
    for m in ("apt-get", "dnf", "pacman", "brew"):
        if shutil.which(m):
            return "apt" if m == "apt-get" else m
    return None


def hint():
    pm = pkg_manager()
    return INSTALL_HINTS.get(pm, "install glslang (or shaderc), Vulkan headers + loader, and a C++20 compiler")


def banner(t):
    print(f"\n=== {t} " + "=" * max(0, 70 - len(t)))


# --------------------------------------------------------------------------
# toolchain discovery
# --------------------------------------------------------------------------
def is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def pm_install(pm, pkgs):
    """Install packages with the system package manager, non-interactively. Returns True on success."""
    if not pkgs or pm not in PKGS:
        return False
    prefix = []
    if pm != "brew" and not is_root():
        if shutil.which("sudo") and sh(["sudo", "-n", "true"]).returncode == 0:
            prefix = ["sudo", "-n"]
        else:
            print("  (not root and passwordless sudo unavailable; skipping package manager)")
            return False
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    if pm == "apt":
        sh(prefix + ["apt-get", "update", "-qq"], env=env)  # tolerate unrelated broken repos
        cmd = prefix + ["apt-get", "install", "-y", "-qq"] + pkgs
    elif pm == "dnf":
        cmd = prefix + ["dnf", "install", "-y"] + pkgs
    elif pm == "pacman":
        cmd = prefix + ["pacman", "-S", "--needed", "--noconfirm"] + pkgs
    else:
        cmd = ["brew", "install"] + pkgs
    print("  running:", " ".join(cmd))
    r = sh(cmd, env=env)
    if r.returncode != 0:
        print("  install failed:", ((r.stderr or r.stdout).strip().splitlines() or ["?"])[-1][:150])
    return r.returncode == 0


def download(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "neon-vulkan-selftest"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_glslang(build: Path):
    """Download a prebuilt glslangValidator from Khronos' main-tot release into build/tools. No root needed."""
    key = GLSLANG_KEYS.get(platform.system())
    mach = platform.machine().lower()
    if not key or (platform.system() == "Linux" and mach not in ("x86_64", "amd64")):
        print(f"  no prebuilt glslang for {platform.system()}/{mach}")
        return None
    tools = build / "tools"
    exe_name = "glslangValidator.exe" if platform.system() == "Windows" else "glslangValidator"
    found = next(tools.rglob(exe_name), None) if tools.exists() else None
    if not found:
        asset = f"glslang-main-{key}-release.zip"
        url = f"{GLSLANG_BASE}/download/main-tot/{asset}"
        try:  # prefer the URL actually listed on the release page (asset names have changed before)
            page = download(f"{GLSLANG_BASE}/expanded_assets/main-tot", 60).decode("utf-8", "replace")
            m = re.search(r'href="([^"]*/main-tot/glslang-main-%s-release\.zip)"' % re.escape(key), page)
            if m:
                url = "https://github.com" + m.group(1)
        except Exception:
            pass
        print("  downloading", url)
        try:
            data = download(url)
        except Exception as e:
            print("  download failed:", e)
            return None
        tools.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tools)
        found = next(tools.rglob(exe_name), None)
        if not found:
            print("  archive did not contain", exe_name)
            return None
    # zip extraction turns the release's glslangValidator -> glslang symlink into a tiny text file; resolve it
    if found.stat().st_size < 256:
        target = found.parent / found.read_text().strip()
        if target.is_file():
            shutil.copyfile(target, found)
    if platform.system() != "Windows":
        found.chmod(found.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    r = sh([str(found), "--version"])
    if r.returncode != 0:
        print("  downloaded glslangValidator won't run here:", (r.stderr or r.stdout).strip()[:120])
        return None
    return str(found)


def fetch_vulkan_headers(build: Path):
    """Download Vulkan-Headers sources (header-only) into build/tools. Returns the include dir or None."""
    inc = build / "tools" / "Vulkan-Headers" / "include"
    if (inc / "vulkan" / "vulkan.h").exists():
        return str(inc)
    print("  downloading", VKHEADERS_URL)
    try:
        data = download(VKHEADERS_URL)
        dest = build / "tools" / "Vulkan-Headers"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            for mem in t.getmembers():
                parts = mem.name.split("/", 1)
                if len(parts) == 2 and parts[1].startswith("include/") and not mem.isdir():
                    mem.name = parts[1]
                    t.extract(mem, dest)
    except Exception as e:
        print("  download failed:", e)
        return None
    return str(inc) if (inc / "vulkan" / "vulkan.h").exists() else None


def discover(args, build: Path):
    tc = {"extra_inc": []}
    tc["glslang"] = args.glslang or which_any("glslangValidator")
    tc["glslc"] = which_any("glslc")
    tc["cxx"] = args.cxx or which_any("g++", "clang++", "c++")
    tools = build / "tools"  # reuse earlier downloads
    if tools.exists():
        if not tc["glslang"]:
            exe = "glslangValidator.exe" if platform.system() == "Windows" else "glslangValidator"
            hit = next(tools.rglob(exe), None)
            tc["glslang"] = str(hit) if hit else None
        inc = tools / "Vulkan-Headers" / "include"
        if (inc / "vulkan" / "vulkan.h").exists():
            tc["extra_inc"].append(str(inc))
    return tc


def check_vulkan_headers(cxx, tmp: Path, extra_inc=()):
    src = tmp / "_hdr_check.cpp"
    src.write_text("#include <vulkan/vulkan.h>\nint main(){VkInstance i=nullptr;(void)i;return 0;}\n")
    cmd = [cxx, "-std=c++20", "-fsyntax-only"] + [f"-I{d}" for d in extra_inc] + [str(src)]
    r = sh(cmd)
    return r.returncode == 0, r.stderr


def ensure_toolchain(args, build: Path):
    """Find everything; install/download whatever is missing. Returns (toolchain, problems)."""
    tc = discover(args, build)
    pm = pkg_manager()
    auto = not args.no_auto_install
    problems = []

    if not tc["cxx"] and not args.shaders_only:
        print("C++ compiler missing.")
        if auto and pm in PKGS and pm_install(pm, PKGS[pm]["cxx"]):
            tc["cxx"] = args.cxx or which_any("g++", "clang++", "c++")
        if not tc["cxx"]:
            problems.append("no C++ compiler" + (" (macOS: run `xcode-select --install`)" if platform.system() == "Darwin" else ""))

    if not (tc["glslang"] or tc["glslc"]):
        print("Shader compiler missing.")
        if auto:
            if pm in PKGS and pm_install(pm, PKGS[pm]["shader"]):
                tc["glslang"] = which_any("glslangValidator")
                tc["glslc"] = which_any("glslc")
            if not (tc["glslang"] or tc["glslc"]):
                print("  falling back to a prebuilt glslang download")
                tc["glslang"] = fetch_glslang(build)
        if not (tc["glslang"] or tc["glslc"]):
            problems.append("no shader compiler (glslangValidator/glslc) and auto-install failed" if auto else "no shader compiler (glslangValidator/glslc)")

    if tc["cxx"] and not args.shaders_only:
        ok, err = check_vulkan_headers(tc["cxx"], build, tc["extra_inc"])
        if not ok:
            print("Vulkan headers missing.")
            if auto:
                if pm in PKGS and pm_install(pm, PKGS[pm]["vkdev"]):
                    ok, err = check_vulkan_headers(tc["cxx"], build, tc["extra_inc"])
                if not ok:
                    print("  falling back to downloading Vulkan-Headers sources")
                    inc = fetch_vulkan_headers(build)
                    if inc:
                        tc["extra_inc"].append(inc)
                        ok, err = check_vulkan_headers(tc["cxx"], build, tc["extra_inc"])
            if not ok:
                problems.append("Vulkan headers not found")
        tc["headers_ok"] = ok
    return tc, problems


def ensure_vulkan_runtime(args):
    """Make sure at least one Vulkan device (real GPU driver or lavapipe) is visible; install runtime pieces if not."""
    devs = device_summary()
    if devs or args.no_auto_install:
        return devs
    pm = pkg_manager()
    if pm in PKGS:
        print("No Vulkan device visible (or vulkaninfo missing); installing the Vulkan runtime/ICD packages.")
        pm_install(pm, PKGS[pm]["runtime"])
        devs = device_summary()
    return devs


def device_summary():
    vi = which_any("vulkaninfo")
    if not vi:
        return None
    env = dict(os.environ, XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    r = sh([vi, "--summary"], env=env)
    names = re.findall(r"deviceName\s*=\s*(.+)", r.stdout)
    return names


# --------------------------------------------------------------------------
# shader compilation
# --------------------------------------------------------------------------
def compile_one(tc, src: Path, out: Path, defines):
    """Try glslang with several target envs, then glslc. Returns (ok, how, log)."""
    log = []
    if tc["glslang"]:
        for env in TARGET_ENVS:
            cmd = [tc["glslang"], "-V", f"--target-env", env] + [f"-D{d}" for d in defines] + [str(src), "-o", str(out)]
            r = sh(cmd)
            if r.returncode == 0 and out.exists():
                return True, f"glslang/{env}", ""
            log.append((r.stdout + r.stderr).strip().splitlines()[-1:] or [""])
    if tc["glslc"]:
        for env in TARGET_ENVS:
            cmd = [tc["glslc"], f"--target-env={env.replace('vulkan', 'vulkan')}"] + [f"-D{d}" for d in defines] + [str(src), "-o", str(out)]
            r = sh(cmd)
            if r.returncode == 0 and out.exists():
                return True, f"glslc/{env}", ""
            log.append((r.stdout + r.stderr).strip().splitlines()[-1:] or [""])
    last = log[-1][0] if log else "no compiler"
    return False, "-", last


def compile_shaders(tc, repo: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sources = sorted(set(list(repo.glob("*.comp")) + list((repo / "shaders").glob("*.comp") if (repo / "shaders").is_dir() else [])))
    rows = []
    for src in sources:
        base = src.stem
        jobs = [("", [])] + VARIANTS.get(base, [])
        for suffix, defs in jobs:
            out = outdir / f"{base}{suffix}.spv"
            if out.exists():
                out.unlink()
            ok, how, err = compile_one(tc, src, out, defs)
            required = (suffix == "") and base in REQUIRED_SHADERS
            rows.append(dict(shader=f"{base}{suffix}", ok=ok, how=how, required=required, err=err))
    return rows, [s.stem for s in sources]


# --------------------------------------------------------------------------
# header patching (matmul pipelines made optional in a COPY)
# --------------------------------------------------------------------------
PATCHES = [
    (r'(\s*)(pipeline_matmul_f16_\s*=\s*build_pipeline_specialized\(dir \+ "matmul_coopmat_f16\.spv"[^;]*;)',
     r'\1if (spirv_file_exists(dir + "matmul_coopmat_f16.spv")) \2'),
    (r'(\s*)(pipeline_matmul_f32_\s*=\s*build_pipeline\(dir \+ "matmul_tiled_f32\.spv"[^;]*;)',
     r'\1if (spirv_file_exists(dir + "matmul_tiled_f32.spv")) \2'),
    (r'(\s*)(pipeline_matmul_q4_0_\s*=\s*build_pipeline\(dir \+ q4_0_variant[^;]*;)',
     r'\1if (spirv_file_exists(dir + q4_0_variant)) \2'),
    (r'(\s*)(pipeline_matmul_q8_0_\s*=\s*build_pipeline\(dir \+ q8_0_variant[^;]*;)',
     r'\1if (spirv_file_exists(dir + q8_0_variant)) \2'),
]


def make_patched_header(repo: Path, build: Path):
    src = (repo / "rawllm_vulkan.hpp").read_text()
    hits = 0
    for pat, rep in PATCHES:
        src, n = re.subn(pat, rep, src, count=1)
        hits += n
    (build / "rawllm_vulkan.hpp").write_text(src)
    return hits


# --------------------------------------------------------------------------
# C++ harness
# --------------------------------------------------------------------------
HARNESS = r'''
// AUTO-GENERATED by neon_vulkan_selftest.py - do not edit
#include "rawllm_vulkan.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>
using namespace vk_backend;
using clk = std::chrono::steady_clock;

static std::mt19937 rng(42);
static void fillv(std::vector<float>& v) { std::uniform_real_distribution<float> d(-1.f, 1.f); for (auto& x : v) x = d(rng); }
static std::string jesc(const std::string& s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') o += ' '; else o += c; } return o; }
static void emit(const std::string& name, const char* status, double err, double tol, const std::string& note, double us = -1, double gflops = -1, double gbs = -1) {
    std::printf("RESULT {\"name\":\"%s\",\"status\":\"%s\",\"err\":%.6g,\"tol\":%.6g,\"us\":%.3f,\"gflops\":%.3f,\"gbs\":%.3f,\"note\":\"%s\"}\n",
                jesc(name).c_str(), status, err, tol, us, gflops, gbs, jesc(note).c_str());
    std::fflush(stdout);
}
static double nerr(const std::vector<float>& y, const std::vector<double>& ref) {
    double mx = 0, mr = 1e-6;
    for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, std::fabs((double)y[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    return mx / mr;
}
static void check(const std::string& name, double err, double tol, const std::string& note = "") {
    bool ok = std::isfinite(err) && err < tol;
    emit(name, ok ? "PASS" : "FAIL", err, tol, note);
}
static void guarded(const std::string& name, const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& e) { emit(name, "ERROR", -1, -1, e.what()); }
}

// ---- fp16 helpers -----------------------------------------------------
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (e <= 0) { if (e < -10) return (uint16_t)sign; m |= 0x800000u; int sh = 14 - e; uint32_t r = m >> sh; if ((m >> (sh - 1)) & 1u) r++; return (uint16_t)(sign | r); }
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t r = sign | ((uint32_t)e << 10) | (m >> 13); if (m & 0x1000u) r++; return (uint16_t)r;
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
    if (e == 0) { if (m == 0) x = s; else { e = 127 - 15 + 1; while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3ffu; x = s | (e << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 127 - 15) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

// ---- quantization helpers (match the shader headers' block layouts) ----
struct QAct { std::vector<int8_t> q; std::vector<float> d; std::vector<int32_t> sum; };
static QAct quant_act(const std::vector<float>& x) {
    size_t nb = x.size() / 32; QAct a; a.q.resize(x.size()); a.d.resize(nb); a.sum.resize(nb);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f; int32_t s = 0;
        for (int j = 0; j < 32; ++j) { int v = (int)std::lround(x[b * 32 + j] * id); v = std::max(-127, std::min(127, v)); a.q[b * 32 + j] = (int8_t)v; s += v; }
        a.d[b] = d; a.sum[b] = s;
    }
    return a;
}
struct QW { std::vector<uint8_t> bytes; std::vector<int8_t> wint; std::vector<float> scale; };  // wint: true integer weights
static QW quant_w4(const std::vector<float>& W, size_t rows, size_t cols) {
    size_t nb = cols / 32; QW w; w.bytes.assign(rows * nb * 18, 0); w.wint.resize(rows * cols); w.scale.resize(rows * nb);
    for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < nb; ++b) {
        const float* p = &W[r * cols + b * 32]; float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(p[j]));
        uint16_t hd = f2h(amax / 7.f); float d = h2f(hd), id = d ? 1.f / d : 0.f; uint8_t* blk = &w.bytes[(r * nb + b) * 18];
        blk[0] = hd & 0xff; blk[1] = hd >> 8; int q[32];
        for (int j = 0; j < 32; ++j) { q[j] = std::max(0, std::min(15, (int)std::lround(p[j] * id) + 8)); w.wint[r * cols + b * 32 + j] = (int8_t)(q[j] - 8); }
        for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)(q[i] | (q[i + 16] << 4));
        w.scale[r * nb + b] = d;
    }
    return w;
}
static QW quant_w8(const std::vector<float>& W, size_t rows, size_t cols) {
    size_t nb = cols / 32; QW w; w.bytes.assign(rows * nb * 34, 0); w.wint.resize(rows * cols); w.scale.resize(rows * nb);
    for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < nb; ++b) {
        const float* p = &W[r * cols + b * 32]; float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(p[j]));
        uint16_t hd = f2h(amax / 127.f); float d = h2f(hd), id = d ? 1.f / d : 0.f; uint8_t* blk = &w.bytes[(r * nb + b) * 34];
        blk[0] = hd & 0xff; blk[1] = hd >> 8;
        for (int j = 0; j < 32; ++j) { int v = std::max(-127, std::min(127, (int)std::lround(p[j] * id))); blk[2 + j] = (uint8_t)(int8_t)v; w.wint[r * cols + b * 32 + j] = (int8_t)v; }
        w.scale[r * nb + b] = d;
    }
    return w;
}
static std::vector<double> qref(const QW& w, const QAct& a, size_t rows, size_t cols) {
    size_t nb = cols / 32; std::vector<double> y(rows, 0.0);
    for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < nb; ++b) {
        long isum = 0; for (int j = 0; j < 32; ++j) isum += (long)w.wint[r * cols + b * 32 + j] * a.q[b * 32 + j];
        y[r] += (double)w.scale[r * nb + b] * (double)a.d[b] * (double)isum;
    }
    return y;
}
static std::vector<double> fref(const std::vector<float>& W, const std::vector<float>& x, size_t rows, size_t cols, bool half_w = false) {
    std::vector<double> y(rows);
    for (size_t r = 0; r < rows; ++r) { double a = 0; for (size_t c = 0; c < cols; ++c) { float w = W[r * cols + c]; if (half_w) w = h2f(f2h(w)); a += (double)w * x[c]; } y[r] = a; }
    return y;
}

int main(int argc, char** argv) {
    std::string shaders = argc > 1 ? argv[1] : "shaders";
    bool full = false, bench = true; int iters = 200;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "--full") full = true; else if (a == "--no-bench") bench = false; else if (a.rfind("--iters=", 0) == 0) iters = std::atoi(a.c_str() + 8); }

    std::unique_ptr<VulkanMatvecBackend> bk;
    try { bk = std::make_unique<VulkanMatvecBackend>(shaders, 16, ""); }
    catch (const std::exception& e) { emit("backend_construct", "ERROR", -1, -1, e.what()); return 2; }
    emit("backend_construct", "PASS", 0, 0, "device accepted all pipelines");
    std::printf("CAPS {\"coopmat\":%d,\"dot4\":%d,\"dp4a_loaded\":%d,\"push_desc\":%d,\"int8_act\":%d,\"int8_loaded\":%d,\"f16\":%d,\"x2\":%d,\"mr1\":%d,\"mr2\":%d,\"mr4\":%d,\"matmul\":%d}\n",
        (int)bk->coop_matrix_supported(), (int)bk->dot_product_supported(), (int)bk->dp4a_pipelines_loaded(), (int)bk->push_descriptor_supported(),
        (int)bk->int8_activation_supported(), (int)bk->int8_activation_pipelines_loaded(), (int)bk->fp16_matvec_available(), (int)bk->matvec_x2_available(),
        (int)bk->matvec_multirow_available(1), (int)bk->matvec_multirow_available(2), (int)bk->matvec_multirow_available(4), HAVE_MATMUL);

    struct Shape { uint32_t rows, cols; };
    std::vector<Shape> shapes = {{64, 128}, {512, 1024}, {77, 1001} /*scalar path, odd rows*/, {3, 36}};
    if (full) shapes.push_back({4096, 4096});

    // 1. F32 matvec (vec4 or scalar path chosen by the backend)
    for (auto s : shapes) guarded("f32_matvec_" + std::to_string(s.rows) + "x" + std::to_string(s.cols), [&] {
        std::vector<float> W((size_t)s.rows * s.cols), x(s.cols), y(s.rows); fillv(W); fillv(x);
        GpuTensor t = bk->upload_weight_f32(W.data(), s.rows, s.cols);
        bk->matvec_f32(t, x.data(), y.data(), s.cols, s.rows);
        check("f32_matvec_" + std::to_string(s.rows) + "x" + std::to_string(s.cols), nerr(y, fref(W, x, s.rows, s.cols)), 1e-4, s.cols % 4 ? "scalar fallback path" : "vec4 path");
        bk->free_tensor(t);
    });

    // 2. Stale-data stress: same slots/buffers reused with NEW x every iteration (exercises persistent mapping + descriptor dedup)
    guarded("f32_reuse_stress", [&] {
        const uint32_t R = 256, C = 512; std::vector<float> Wa(R * C), Wb(R * C); fillv(Wa); fillv(Wb);
        GpuTensor ta = bk->upload_weight_f32(Wa.data(), R, C), tb = bk->upload_weight_f32(Wb.data(), R, C);
        double worst = 0;
        for (int it = 0; it < 64; ++it) {
            std::vector<float> x(C), ya(R), yb(R); fillv(x);
            bk->begin_batch(); bk->queue_matvec_f32(ta, x.data(), ya.data(), C, R); bk->queue_matvec_f32(tb, x.data(), yb.data(), C, R); bk->wait_batch(bk->end_batch());
            worst = std::max({worst, nerr(ya, fref(Wa, x, R, C)), nerr(yb, fref(Wb, x, R, C))});
        }
        check("f32_reuse_stress", worst, 1e-4, "64 iterations, 2 ops/batch, fresh x each time");
        bk->free_tensor(ta); bk->free_tensor(tb);
    });

    // 3. One batch with 7 distinct projections (Q,K,V,O,gate,up,down shape)
    guarded("f32_batch7", [&] {
        const uint32_t R = 128, C = 256; std::vector<std::vector<float>> Ws(7, std::vector<float>(R * C)); std::vector<GpuTensor> ts;
        for (auto& w : Ws) { fillv(w); ts.push_back(bk->upload_weight_f32(w.data(), R, C)); }
        std::vector<float> x(C); fillv(x); std::vector<std::vector<float>> ys(7, std::vector<float>(R));
        bk->begin_batch(); for (int i = 0; i < 7; ++i) bk->queue_matvec_f32(ts[i], x.data(), ys[i].data(), C, R); bk->wait_batch(bk->end_batch());
        double worst = 0; for (int i = 0; i < 7; ++i) worst = std::max(worst, nerr(ys[i], fref(Ws[i], x, R, C)));
        check("f32_batch7", worst, 1e-4, "7 dispatches in one submit");
        for (auto& t : ts) bk->free_tensor(t);
    });

    // 4. Experimental F32 variants
    if (bk->matvec_x2_available()) for (uint32_t R : {128u, 129u}) guarded("f32_x2_" + std::to_string(R), [&] {
        const uint32_t C = 512; std::vector<float> W((size_t)R * C), x(C), y(R); fillv(W); fillv(x);
        GpuTensor t = bk->upload_weight_f32(W.data(), R, C); bk->matvec_f32_x2(t, x.data(), y.data(), C, R);
        check("f32_x2_" + std::to_string(R), nerr(y, fref(W, x, R, C)), 1e-4); bk->free_tensor(t);
    }); else emit("f32_x2", "SKIP", 0, 0, "matvec_f32_x2.spv not built");
    for (uint32_t rps : {1u, 2u, 4u}) {
        std::string nm = "f32_multirow" + std::to_string(rps);
        if (!bk->matvec_multirow_available(rps)) { emit(nm, "SKIP", 0, 0, "matvec_f32_multirow.spv not built"); continue; }
        guarded(nm, [&] {
            const uint32_t R = 130, C = 512; std::vector<float> W((size_t)R * C), x(C), y(R); fillv(W); fillv(x);
            GpuTensor t = bk->upload_weight_f32(W.data(), R, C); bk->matvec_f32_multirow(rps, t, x.data(), y.data(), C, R);
            check(nm, nerr(y, fref(W, x, R, C)), 1e-4); bk->free_tensor(t);
        });
    }

    // 5. F16 weights
    if (bk->fp16_matvec_available()) guarded("f16_matvec", [&] {
        const uint32_t R = 200, C = 512; std::vector<float> W((size_t)R * C), x(C), y(R); fillv(W); fillv(x);
        GpuTensor t = bk->upload_weight_f16(W.data(), R, C); bk->matvec_f16(t, x.data(), y.data(), C, R);
        check("f16_matvec", nerr(y, fref(W, x, R, C, true)), 2e-3, "reference uses fp16-rounded weights"); bk->free_tensor(t);
    }); else emit("f16_matvec", "SKIP", 0, 0, "matvec_f16.spv not built");

    // 6. Quantized GEMV
    for (int bits : {4, 8}) {
        std::string nm = std::string("q") + (bits == 4 ? "4_0" : "8_0") + "_matvec";
        guarded(nm, [&] {
            const uint32_t R = 192, C = 1024; std::vector<float> W((size_t)R * C), x(C), y(R); fillv(W); fillv(x);
            QW qw = bits == 4 ? quant_w4(W, R, C) : quant_w8(W, R, C); QAct qa = quant_act(x);
            GpuTensor t = bits == 4 ? bk->upload_weight_q4_0(qw.bytes.data(), R, C) : bk->upload_weight_q8_0(qw.bytes.data(), R, C);
            if (bits == 4) bk->matvec_q4_0(t, qa.d.data(), qa.sum.data(), qa.q.data(), y.data(), C, R);
            else           bk->matvec_q8_0(t, qa.d.data(), qa.sum.data(), qa.q.data(), y.data(), C, R);
            check(nm, nerr(y, qref(qw, qa, R, C)), 1e-3, bk->int8_activation_pipelines_loaded() ? "int8-activation shader" : "portable int32-activation shader");
            bk->free_tensor(t);
        });
    }

    // 7. F32 GEMM (only if its shader was built)
#if HAVE_MATMUL
    for (auto s : std::vector<std::array<uint32_t, 3>>{{37, 40, 29}, {1, 64, 64}, {17, 17, 17}}) {
        std::string nm = "matmul_f32_" + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "x" + std::to_string(s[2]);
        guarded(nm, [&] {
            uint32_t M = s[0], K = s[1], N = s[2]; std::vector<float> B((size_t)K * N), A((size_t)M * K), Y((size_t)M * N); fillv(B); fillv(A);
            GpuTensor t = bk->upload_weight_f32(B.data(), K, N); bk->begin_batch(); bk->queue_matmul_f32(t, A.data(), Y.data(), M, K, N); bk->wait_batch(bk->end_batch());
            std::vector<double> ref((size_t)M * N); for (uint32_t m = 0; m < M; ++m) for (uint32_t n = 0; n < N; ++n) { double a = 0; for (uint32_t k = 0; k < K; ++k) a += (double)A[m * K + k] * B[k * N + n]; ref[m * N + n] = a; }
            check(nm, nerr(Y, ref), 1e-4); bk->free_tensor(t);
        });
    }
#else
    emit("matmul_f32", "SKIP", 0, 0, "no matmul_tiled_f32.comp in repo, GEMM path not testable");
#endif

    // 8. Benchmarks
    if (bench) {
        auto timeit = [&](const std::string& nm, double flops, double bytes, const std::function<void()>& fn) {
            guarded(nm, [&] {
                for (int i = 0; i < std::min(20, iters); ++i) fn();
                auto t0 = clk::now(); for (int i = 0; i < iters; ++i) fn(); auto t1 = clk::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
                emit(nm, "BENCH", 0, 0, std::to_string(iters) + " timed iters", us, flops / (us * 1e-6) / 1e9, bytes / (us * 1e-6) / 1e9);
            });
        };
        const uint32_t N = full ? 4096 : 2048;
        std::vector<float> W((size_t)N * N), x(N), y(N); fillv(W); fillv(x);
        GpuTensor tf = bk->upload_weight_f32(W.data(), N, N);
        timeit("bench_f32_blocking_" + std::to_string(N), 2.0 * N * N, 4.0 * N * N, [&] { bk->matvec_f32(tf, x.data(), y.data(), N, N); });
        { std::vector<GpuTensor> ts; std::vector<std::vector<float>> ys(7, std::vector<float>(N)); for (int i = 0; i < 7; ++i) ts.push_back(bk->upload_weight_f32(W.data(), N, N));
          timeit("bench_f32_batch7_" + std::to_string(N), 7 * 2.0 * N * N, 7 * 4.0 * N * N, [&] { bk->begin_batch(); for (int i = 0; i < 7; ++i) bk->queue_matvec_f32(ts[i], x.data(), ys[i].data(), N, N); bk->wait_batch(bk->end_batch()); });
          for (auto& t : ts) bk->free_tensor(t); }
        QAct qa = quant_act(x); QW q4 = quant_w4(W, N, N), q8 = quant_w8(W, N, N);
        GpuTensor t4 = bk->upload_weight_q4_0(q4.bytes.data(), N, N), t8 = bk->upload_weight_q8_0(q8.bytes.data(), N, N);
        timeit("bench_q4_0_blocking_" + std::to_string(N), 2.0 * N * N, (double)q4.bytes.size(), [&] { bk->matvec_q4_0(t4, qa.d.data(), qa.sum.data(), qa.q.data(), y.data(), N, N); });
        timeit("bench_q8_0_blocking_" + std::to_string(N), 2.0 * N * N, (double)q8.bytes.size(), [&] { bk->matvec_q8_0(t8, qa.d.data(), qa.sum.data(), qa.q.data(), y.data(), N, N); });
        if (bk->matvec_x2_available()) timeit("bench_f32_x2_" + std::to_string(N), 2.0 * N * N, 4.0 * N * N, [&] { bk->matvec_f32_x2(tf, x.data(), y.data(), N, N); });
        if (bk->matvec_multirow_available(4)) timeit("bench_f32_multirow4_" + std::to_string(N), 2.0 * N * N, 4.0 * N * N, [&] { bk->matvec_f32_multirow(4, tf, x.data(), y.data(), N, N); });
        bk->free_tensor(tf); bk->free_tensor(t4); bk->free_tensor(t8);
    }
    return 0;
}
'''


def build_harness(tc, repo: Path, build: Path, have_matmul: bool):
    cpp = build / "vk_selftest_harness.cpp"
    cpp.write_text(HARNESS.replace("#include <vector>\nusing namespace", "#include <vector>\n#include <array>\nusing namespace", 1))
    exe = build / ("vk_selftest_harness.exe" if platform.system() == "Windows" else "vk_selftest_harness")
    cmd = [tc["cxx"], "-std=c++20", "-O2", "-pthread", "-DUSE_VULKAN", f"-DHAVE_MATMUL={1 if have_matmul else 0}",
           "-I", str(build), "-I", str(repo)] + [f"-I{d}" for d in tc.get("extra_inc", [])] + [str(cpp), "-o", str(exe)]
    if platform.system() == "Darwin":
        cmd += ["-lvulkan"]
    else:
        cmd += ["-lvulkan"]
    r = sh(cmd)
    return r.returncode == 0, exe, (r.stderr or r.stdout)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--build-dir", default="build_vk_selftest")
    ap.add_argument("--glslang", help="path to glslangValidator")
    ap.add_argument("--cxx", help="C++ compiler")
    ap.add_argument("--full", action="store_true", help="include 4096x4096 correctness + bench at 4096")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--shaders-only", action="store_true", help="only compile shaders and report")
    ap.add_argument("--lavapipe", action="store_true", help="force Mesa lavapipe (CPU Vulkan) via VK_ICD_FILENAMES")
    ap.add_argument("--no-auto-install", action="store_true", help="don't install/download missing tools; just report them")
    ap.add_argument("--install-deps", action="store_true", help=argparse.SUPPRESS)  # legacy, now the default behaviour
    ap.add_argument("--report", default="vulkan_selftest_report.json")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    build = Path(args.build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)

    banner("environment")
    tc, problems = ensure_toolchain(args, build)
    print("glslangValidator:", tc["glslang"] or "NOT FOUND")
    print("glslc           :", tc["glslc"] or "not found")
    print("C++ compiler    :", tc["cxx"] or "NOT FOUND")
    if tc["cxx"] and not args.shaders_only:
        print("Vulkan headers  :", "ok" + (f" (downloaded to {tc['extra_inc'][-1]})" if tc["extra_inc"] else "") if tc.get("headers_ok") else "NOT FOUND")
    if problems:
        print("\nCould not set up:", "; ".join(problems), "\nInstall manually with:\n  " + hint())
        return 3

    if not (repo / "rawllm_vulkan.hpp").exists():
        print(f"\nrawllm_vulkan.hpp not found in {repo}. Run from the repo root, or pass --repo.")
        return 3
    if not list(repo.glob("*.comp")) and not (repo / "shaders").is_dir():
        print("\nNo .comp shader sources here. They only exist on the VULKAN-EXPERIMENTAL branch:\n"
              "  git fetch origin VULKAN-EXPERIMENTAL && git checkout VULKAN-EXPERIMENTAL")
        return 3

    devs = ensure_vulkan_runtime(args)
    print("Vulkan devices  :", ", ".join(d.strip() for d in devs) if devs else "(none found: the backend will fail to construct)")

    if args.lavapipe:
        for p in ("/usr/share/vulkan/icd.d/lvp_icd.json", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"):
            if Path(p).exists():
                os.environ["VK_ICD_FILENAMES"] = p
                print("Forcing lavapipe:", p)
                break
        else:
            print("lavapipe ICD json not found; install mesa-vulkan-drivers")

    banner("compiling shaders")
    shader_dir = build / "shaders"
    rows, sources = compile_shaders(tc, repo, shader_dir)
    for r in rows:
        tag = "OK  " if r["ok"] else ("FAIL" if r["required"] else "skip")
        print(f"  [{tag}] {r['shader']:<28} {r['how'] if r['ok'] else r['err'][:70]}")
    req_fail = [r["shader"] for r in rows if r["required"] and not r["ok"]]
    for need in REQUIRED_SHADERS:
        if need not in sources:
            req_fail.append(f"{need} (source missing)")
    if req_fail:
        print("\nRequired shader(s) failed:", ", ".join(req_fail))
        return 4
    report = dict(shaders=rows, tests=[], caps={}, devices=devs, patched_header=False)
    if args.shaders_only:
        Path(args.report).write_text(json.dumps(report, indent=2))
        return 0

    banner("preparing backend header")
    have = lambda n: (shader_dir / f"{n}.spv").exists()
    q_ok = lambda q: have(q) or have(q + "_dp4a")
    have_matmul = have("matmul_tiled_f32") and q_ok("matmul_tiled_q4_0") and q_ok("matmul_tiled_q8_0")
    if have_matmul:
        shutil.copy(repo / "rawllm_vulkan.hpp", build / "rawllm_vulkan.hpp")
        print("All GEMM shaders present: using the original rawllm_vulkan.hpp")
    else:
        n = make_patched_header(repo, build)
        report["patched_header"] = True
        print("matmul_*.comp sources/SPIR-V missing, but rawllm_vulkan.hpp's constructor builds those pipelines\n"
              f"unconditionally and would throw. Using a PATCHED COPY in {build.name}/ ({n}/4 sites made optional).\n"
              "Your repo files are untouched.")
        if n < 4:
            print("  warning: header layout changed; some sites weren't patched, construction may still fail.")

    banner("building harness")
    ok, exe, log = build_harness(tc, repo, build, have_matmul and have("matmul_tiled_f32"))
    if not ok:
        print(log[-3000:])
        print("\nBuild failed. If this is a missing -lvulkan, the Vulkan loader library is not installed;\n"
              "headers can be downloaded but the loader needs a package:\n  " + hint())
        return 5
    print("built", exe.name)

    banner("running")
    cmd = [str(exe), str(shader_dir)] + (["--full"] if args.full else []) + (["--no-bench"] if args.no_bench else []) + [f"--iters={args.iters}"]
    t0 = time.time()
    proc = sh(cmd, cwd=build)
    results = []
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            results.append(json.loads(line[7:]))
        elif line.startswith("CAPS "):
            report["caps"] = json.loads(line[5:])
    if proc.returncode not in (0, 2) or (proc.returncode != 0 and not results):
        print("harness crashed (exit %d):\n%s" % (proc.returncode, (proc.stderr or "")[-2000:]))
    report["tests"] = results

    if report["caps"]:
        c = report["caps"]
        print("device capabilities:", ", ".join(f"{k}={v}" for k, v in c.items()))
    print(f"\n{'test':<34}{'status':<7}{'err':>11}{'tol':>9}  note")
    fails = 0
    for r in results:
        if r["status"] == "BENCH":
            continue
        fails += r["status"] in ("FAIL", "ERROR")
        err = f"{r['err']:.2e}" if r["err"] >= 0 else "-"
        tol = f"{r['tol']:.0e}" if r["tol"] > 0 else "-"
        print(f"{r['name']:<34}{r['status']:<7}{err:>11}{tol:>9}  {r['note'][:60]}")
    bn = [r for r in results if r["status"] == "BENCH"]
    if bn:
        print(f"\n{'benchmark':<34}{'us/call':>10}{'GFLOP/s':>10}{'GB/s':>9}")
        for r in bn:
            print(f"{r['name']:<34}{r['us']:>10.1f}{r['gflops']:>10.1f}{r['gbs']:>9.1f}")
        if devs and any("llvmpipe" in d or "lavapipe" in d.lower() for d in devs):
            print("(device is a software rasterizer: timings are meaningless, only correctness counts)")
    passed = sum(r["status"] == "PASS" for r in results)
    skipped = sum(r["status"] == "SKIP" for r in results)
    print(f"\n{passed} passed, {fails} failed, {skipped} skipped  ({time.time() - t0:.1f}s)")
    Path(args.report).write_text(json.dumps(report, indent=2))
    print("report ->", args.report)
    return 1 if (fails or proc.returncode not in (0,)) else 0


if __name__ == "__main__":
    sys.exit(main())
