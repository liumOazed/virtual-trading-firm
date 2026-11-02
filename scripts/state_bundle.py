"""
state_bundle.py
===============
Moves the trading firm's runtime state between your PC and GitHub Actions.

Why: the models (*.pkl), price caches, trade/equity logs and growth-book CSVs
are git-ignored, so a fresh GitHub checkout has none of them. This script packs
them into ONE encrypted bundle and stores it on a separate `bot-state` branch.
The repo is public, so the bundle is encrypted (Fernet / AES-128 + HMAC) with a
key that only lives on your PC and in the STATE_KEY repo secret.

The `bot-state` branch always holds a single commit (force-pushed), so the repo
does not grow by a full bundle every day.

Commands (run from the repo root):
  python scripts/state_bundle.py keygen          # print a new key -> STATE_KEY secret
  python scripts/state_bundle.py list            # show which files would be bundled
  python scripts/state_bundle.py push            # PC -> GitHub (seed / hand back state)
  python scripts/state_bundle.py pull            # GitHub -> PC (get the latest logs)

  # used by the workflow:
  python scripts/state_bundle.py fetch   --dest DIR   # download bot-state branch
  python scripts/state_bundle.py unpack  --src DIR    # decrypt into the checkout
  python scripts/state_bundle.py pack    --out DIR --no-pins
  python scripts/state_bundle.py publish --src DIR    # force-push DIR as bot-state

The key is read from the STATE_KEY env var, or from a git-ignored file
`.state_key` in the repo root (keygen offers to write it).

Install: pip install cryptography
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "bot-state"
KEY_FILE = ROOT / ".state_key"
CHUNK = 90 * 1024 * 1024          # stay under GitHub's 100 MB per-file limit
BUNDLE_PREFIX = "state.bin."

# Directories whose git-ignored files make up the runtime state.
STATE_DIRS = [
    "3_market_data/cache",
    "4_signals",
    "5_backtesting/results",
    "8_live_trading/data",
    "9_aria_growth",
]
# Tracked files the daily run rewrites (kept in the bundle so the bot's copy wins).
TRACKED_STATE = [
    "8_live_trading/data/regime_state.json",
]
# Never bundle secrets, environments or caches.
EXCLUDE_NAMES = {".env", "desktop.ini", ".DS_Store", "Thumbs.db"}
EXCLUDE_PARTS = {"__pycache__", ".venv", "venv", "env", ".git", "test", "eval_logs", "tb_logs"}
EXCLUDE_SUFFIXES = {".env", ".pyc", ".pyo", ".ini", ".tmp", ".bak", ".log"}

# Packages whose versions must match the PC that trained/pickled the models.
PIN_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "hmmlearn",
    "yfinance", "pandas-ta", "pandas_ta", "optuna", "shap", "requests",
    "python-dotenv", "pytz", "feedparser", "openpyxl", "newsapi-python",
    "langchain-core", "langchain_core", "matplotlib",
]


# ── helpers ──────────────────────────────────────────────────────────────────
def git(*args, input_bytes=None, check=True) -> bytes:
    r = subprocess.run(["git", *args], cwd=ROOT, input=input_bytes,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.decode(errors='replace')}")
    return r.stdout


def load_key() -> bytes:
    k = os.environ.get("STATE_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    if not k:
        raise SystemExit("No STATE_KEY env var and no .state_key file. Run: "
                         "python scripts/state_bundle.py keygen")
    return k.encode()


def fernet():
    from cryptography.fernet import Fernet
    return Fernet(load_key())


def excluded(rel: str) -> bool:
    p = Path(rel)
    if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return any(part in EXCLUDE_PARTS for part in p.parts)


def state_files() -> list:
    """Git-ignored (plus listed tracked) files under STATE_DIRS that exist on disk."""
    out = git("ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", *STATE_DIRS)
    files = {f for f in out.decode().split("\0") if f}
    files.update(f for f in TRACKED_STATE if (ROOT / f).exists())
    return sorted(f for f in files if (ROOT / f).is_file() and not excluded(f))


def make_tar(files) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            tar.add(str(ROOT / f), arcname=f)
    return buf.getvalue()


def safe_extract(tar_bytes: bytes):
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = (ROOT / m.name).resolve()
            if not str(target).startswith(str(ROOT)) or not (m.isfile() or m.isdir()):
                raise SystemExit(f"Refusing suspicious bundle entry: {m.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(ROOT, filter="data")
        else:
            tar.extractall(ROOT)
        return [m.name for m in tar.getmembers() if m.isfile()]


def pinned_requirements() -> str:
    from importlib import metadata
    lines, seen = [], set()
    for name in PIN_PACKAGES:
        try:
            v = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        canon = name.replace("_", "-").lower()
        if canon not in seen:
            seen.add(canon)
            lines.append(f"{canon}=={v}")
    return "\n".join(lines) + "\n"


# ── pack / unpack (directory form) ──────────────────────────────────────────
def pack(out_dir: Path, files=None, write_pins=True) -> dict:
    files = files if files is not None else state_files()
    if not files:
        raise SystemExit("No state files found — nothing to bundle.")
    blob = fernet().encrypt(make_tar(files))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(BUNDLE_PREFIX + "*"):
        old.unlink()
    n = 0
    for i in range(0, len(blob), CHUNK):
        (out_dir / f"{BUNDLE_PREFIX}{n:03d}").write_bytes(blob[i:i + CHUNK])
        n += 1
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(files), "n_chunks": n, "encrypted_bytes": len(blob),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if write_pins:
        (out_dir / "requirements.lock").write_text(pinned_requirements())
        (out_dir / "python-version").write_text(
            f"{sys.version_info.major}.{sys.version_info.minor}\n")
    return manifest


def unpack(src_dir: Path) -> list:
    chunks = sorted(src_dir.glob(BUNDLE_PREFIX + "*"))
    if not chunks:
        raise SystemExit(f"No {BUNDLE_PREFIX}* chunks in {src_dir}")
    blob = b"".join(c.read_bytes() for c in chunks)
    from cryptography.fernet import InvalidToken
    try:
        tar_bytes = fernet().decrypt(blob)
    except InvalidToken:
        raise SystemExit("Could not decrypt the state bundle — STATE_KEY does not match.")
    return safe_extract(tar_bytes)


# ── branch transport (used from your PC) ────────────────────────────────────
def push_dir(src_dir: Path, message: str):
    """Commit every file in src_dir as the single commit of BRANCH and force-push it."""
    entries = []
    for p in sorted(src_dir.iterdir()):
        sha = git("hash-object", "-w", str(p)).decode().strip()
        entries.append(f"100644 blob {sha}\t{p.name}")
    tree = git("mktree", input_bytes=("\n".join(entries) + "\n").encode()).decode().strip()
    commit = git("commit-tree", tree, "-m", message).decode().strip()
    git("push", "--force", "origin", f"{commit}:refs/heads/{BRANCH}")
    return commit


def fetch_dir(dest: Path) -> bool:
    r = subprocess.run(["git", "fetch", "--depth=1", "origin", BRANCH], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    listing = git("ls-tree", "--name-only", "FETCH_HEAD").decode().split()
    for name in listing:
        (dest / name).write_bytes(git("cat-file", "blob", f"FETCH_HEAD:{name}"))
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Encrypted runtime-state bundle for GitHub Actions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen")
    sub.add_parser("list")
    sub.add_parser("push")
    sub.add_parser("pull")
    p = sub.add_parser("pack"); p.add_argument("--out", required=True)
    p.add_argument("--no-pins", action="store_true",
                   help="don't rewrite requirements.lock / python-version (CI keeps the PC's pins)")
    u = sub.add_parser("unpack"); u.add_argument("--src", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--dest", required=True)
    b = sub.add_parser("publish"); b.add_argument("--src", required=True)
    b.add_argument("--message", default="state from GitHub Actions")
    args = ap.parse_args()

    if args.cmd == "keygen":
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        print(key)
        if not KEY_FILE.exists():
            KEY_FILE.write_text(key + "\n")
            print(f"\nSaved to {KEY_FILE.name} (git-ignored). Add the same value as the "
                  f"STATE_KEY repository secret.", file=sys.stderr)
        else:
            print(f"\n{KEY_FILE.name} already exists — not overwritten.", file=sys.stderr)

    elif args.cmd == "list":
        files = state_files()
        total = 0
        for f in files:
            size = (ROOT / f).stat().st_size
            total += size
            print(f"{size / 1e6:9.2f} MB  {f}")
        print(f"\n{len(files)} files, {total / 1e6:.1f} MB before compression")

    elif args.cmd == "push":
        with tempfile.TemporaryDirectory() as d:
            m = pack(Path(d))
            commit = push_dir(Path(d), f"state from PC {m['created_utc']} ({m['n_files']} files)")
        print(f"Pushed {m['n_files']} files ({m['encrypted_bytes'] / 1e6:.1f} MB encrypted) "
              f"to branch {BRANCH} @ {commit[:8]}")

    elif args.cmd == "pull":
        with tempfile.TemporaryDirectory() as d:
            if not fetch_dir(Path(d)):
                raise SystemExit(f"No {BRANCH} branch on origin yet.")
            names = unpack(Path(d))
        print(f"Restored {len(names)} files from {BRANCH}.")

    elif args.cmd == "pack":
        m = pack(Path(args.out), write_pins=not args.no_pins)
        print(f"Packed {m['n_files']} files into {m['n_chunks']} chunk(s), "
              f"{m['encrypted_bytes'] / 1e6:.1f} MB encrypted")

    elif args.cmd == "fetch":
        if not fetch_dir(Path(args.dest)):
            raise SystemExit(f"No {BRANCH} branch on origin yet — seed it from your PC with "
                             f"`python scripts/state_bundle.py push`.")
        print(f"Fetched {BRANCH} into {args.dest}")

    elif args.cmd == "publish":
        commit = push_dir(Path(args.src), args.message)
        print(f"Published {args.src} to {BRANCH} @ {commit[:8]}")

    elif args.cmd == "unpack":
        names = unpack(Path(args.src))
        print(f"Restored {len(names)} state files")


if __name__ == "__main__":
    main()
