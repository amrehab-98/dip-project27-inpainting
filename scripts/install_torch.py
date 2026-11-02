"""Install the PyTorch build that matches this machine's GPU.

No single PyTorch wheel supports every GPU generation, so torch is not pinned in
requirements.txt. Run this script first, then install the requirements:

    python scripts/install_torch.py
    python -m pip install -r requirements.txt

The multi-GB wheels are downloaded in small resumable chunks into .cache/wheels/,
verified against the index's SHA-256, then installed from the local files. If the
connection drops, rerun the script and it continues where it stopped.
"""

import argparse
import hashlib
import http.client
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TORCH_VERSION = "2.9.1"
TORCHVISION_VERSION = "0.24.1"
INDEX_URL = "https://download.pytorch.org/whl/{variant}"
VARIANTS = ("cu128", "cu126", "cpu")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "wheels"
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError)


def lowest_compute_capability() -> float | None:
    """Return the oldest NVIDIA GPU's compute capability, or None without one."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    capabilities = [float(line) for line in output.split() if line.strip()]
    return min(capabilities) if capabilities else None


def choose_variant() -> str:
    """cu128 covers Volta (7.0) through Blackwell (12.x); cu126 still covers older GPUs."""
    capability = lowest_compute_capability()
    if capability is None:
        return "cpu"
    return "cu128" if capability >= 7.0 else "cu126"


def find_wheel(package: str, version: str, variant: str) -> tuple[str, str, str]:
    """Return (url, filename, sha256) of the wheel for this Python and platform."""
    index = urllib.request.urlopen(f"{INDEX_URL.format(variant=variant)}/{package}/",
                                   timeout=60).read().decode()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    prefix = f"{package}-{version}%2B{variant}-{python_tag}-{python_tag}-"
    machine = platform.machine().lower()
    for href in re.findall(r'href="([^"]+)"', index):
        url, _, digest = href.partition("#sha256=")
        filename = url.rsplit("/", 1)[-1]
        if not filename.startswith(prefix):
            continue
        tag = filename[len(prefix):-len(".whl")]
        if (sys.platform == "win32" and tag == f"win_{machine}") or \
                (sys.platform.startswith("linux") and "linux" in tag and tag.endswith(machine)):
            return url, filename.replace("%2B", "+"), digest
    raise SystemExit(f"No {package} {version}+{variant} wheel for {python_tag} on "
                     f"{sys.platform}/{machine}; try another --variant.")


def fetch_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    """Fetch bytes [start, end] and return them with the file's total size."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise OSError(f"server ignored the range request (HTTP {response.status})")
        data = response.read()
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    if len(data) != end - start + 1 and start + len(data) != total:
        raise OSError("chunk ended early")
    return data, total


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str, chunk_bytes: int, max_retries: int) -> None:
    """Download url to dest chunk by chunk, resuming from any earlier partial file."""
    if dest.exists() and sha256_of(dest) == sha256:
        print(f"Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    total, failures = None, 0
    print(f"Downloading {dest.name}" + (f" (resuming at {done / 2**20:.0f} MB)" if done else ""))
    while total is None or done < total:
        end = done + chunk_bytes - 1 if total is None else min(done + chunk_bytes, total) - 1
        try:
            data, total = fetch_range(url, done, end)
        except NETWORK_ERRORS as error:
            failures += 1
            if failures > max_retries:
                raise SystemExit(f"\nGave up after {max_retries} retries in a row ({error}). "
                                 "Rerun the script to resume.")
            wait = min(2 ** failures, 30)
            print(f"\n  chunk failed ({error}); retry {failures}/{max_retries} in {wait}s")
            time.sleep(wait)
            continue
        with part.open("ab") as file:
            file.write(data)
        done += len(data)
        failures = 0
        print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB ({100 * done / total:.1f}%)",
              end="", flush=True)
    print()
    if sha256_of(part) != sha256:
        part.unlink()
        raise SystemExit(f"Checksum mismatch for {dest.name}; deleted it, rerun to download again.")
    for _ in range(10):
        # Antivirus scanners can briefly lock a freshly written file on Windows.
        try:
            part.replace(dest)
            return
        except PermissionError:
            time.sleep(2)
    part.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS,
                        help="override detection (default: chosen from nvidia-smi)")
    parser.add_argument("--chunk-mb", type=int, default=8, help="download chunk size (default 8)")
    parser.add_argument("--retries", type=int, default=50,
                        help="retries per chunk before giving up (default 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selected wheels and pip command only")
    args = parser.parse_args()

    if sys.platform == "darwin":
        # macOS wheels come from PyPI and include Apple MPS support.
        print("Selected PyTorch build: macos")
        command = [sys.executable, "-m", "pip", "install",
                   f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
        return

    variant = args.variant or choose_variant()
    print(f"Selected PyTorch build: {variant}")
    wheels = [find_wheel("torch", TORCH_VERSION, variant),
              find_wheel("torchvision", TORCHVISION_VERSION, variant)]
    for url, filename, sha256 in wheels:
        print(f"  {url}")
        if not args.dry_run:
            download(url, CACHE_DIR / filename, sha256, args.chunk_mb * 2**20, args.retries)
    # Small dependencies (sympy, networkx, ...) still come from the PyTorch index.
    command = [sys.executable, "-m", "pip", "install",
               *(str(CACHE_DIR / filename) for _, filename, _ in wheels),
               "--index-url", INDEX_URL.format(variant=variant)]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
