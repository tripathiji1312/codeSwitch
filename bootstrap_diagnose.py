#!/usr/bin/env python3
"""bootstrap_diagnose.py — one-command recovery + audio diagnostic for Kaggle.

A fresh Kaggle session has neither the notebook's dependency stack nor the
``/kaggle/working`` artifacts.  This script:

  1. finds (or clones) the repo,
  2. installs the exact dependency stack the notebook deps cell uses
     (parler-tts, transformers==4.46.1, protobuf==5.29.5, ...),
  3. restores the student checkpoints and (optionally) the teacher cache from
     the W&B artifacts written by ``push_artifacts_to_wandb.py``,
  4. runs ``diagnose_audio_bug.py`` in a *fresh* subprocess, so the newly
     installed packages are imported cleanly even if this notebook kernel is old.

Typical use (from anywhere; the script cd's to the repo):

    !python bootstrap_diagnose.py

Useful variants:

    # only the (small) student checkpoint artifact; skip the 2000-shard cache
    !python bootstrap_diagnose.py --groups student

    # pull the final zip too and unpack the eval audio/results
    !python bootstrap_diagnose.py --groups student,teacher,bundle --unzip-bundle

    # artifact lives under a non-default entity/project
    !python bootstrap_diagnose.py --entity my-entity --project codeswitch-tts
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/tripathiji1312/codeSwitch.git"
DEFAULT_REPO = Path("/kaggle/working/codeSwitch")
DEFAULT_PROJECT = "codeswitch-tts"
DEST = {"student": ".", "teacher": ".", "bundle": "/kaggle/working"}


# --------------------------------------------------------------------------
# repo
# --------------------------------------------------------------------------
def find_repo(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [DEFAULT_REPO, Path("codeSwitch"), Path(".")]
    for cand in candidates:
        if cand and (cand / "run_full_evaluation.py").is_file():
            return cand.resolve()
    return None


def clone_repo(dest: Path) -> Path:
    print(f"[bootstrap] cloning {REPO_URL} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(dest)], check=True)
    return dest


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------
PIP_STEPS = [
    ["-U", "git+https://github.com/huggingface/parler-tts.git"],
    ["transformers==4.46.1"],
    ["descript-audio-codec", "soundfile", "scipy", "accelerate", "sentencepiece"],
    # Notebook parity: the eval-suite extras ride along so a later eval in this
    # session does not reinstall (and possibly re-break) the stack.
    ["jiwer", "matplotlib", "seaborn", "pandas"],
    ["protobuf==5.29.5"],
    ["wandb"],
]


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except Exception:
        return None


def dependency_problems() -> list[str]:
    problems: list[str] = []
    try:
        have_parler = importlib.util.find_spec("parler_tts") is not None
    except Exception:
        have_parler = False
    if not have_parler:
        problems.append("parler_tts missing")
    tv = _version("transformers")
    if tv != "4.46.1":
        problems.append(f"transformers={tv} (need 4.46.1)")
    pv = _version("protobuf")
    try:
        pv_ok = pv is not None and tuple(map(int, pv.split(".")[:2])) >= (5, 26)
    except Exception:
        pv_ok = False
    if not pv_ok:
        problems.append(f"protobuf={pv} (need >= 5.26)")
    for pkg in ("torch", "soundfile", "scipy", "sentencepiece", "accelerate",
                "jiwer", "matplotlib", "wandb"):
        if _version(pkg) is None:
            problems.append(f"{pkg} missing")
    return problems


def run_pip(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("[bootstrap] " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"pip step failed ({rc}): {' '.join(cmd)}")


def install_dependencies() -> None:
    print("[bootstrap] installing the notebook dependency stack "
          "(this can take 3-6 minutes)...")
    for step in PIP_STEPS:
        run_pip(["install", "-q", *step])
    print("[bootstrap] dependency install finished")


def test_import(module: str) -> bool:
    code = f"import {module}; print({module}.__file__)"
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[bootstrap] import {module} FAILED")
        print((proc.stdout + proc.stderr).strip()[-3000:])
        return False
    print(f"[bootstrap] import {module} OK: {proc.stdout.strip()}")
    return True


# --------------------------------------------------------------------------
# W&B artifacts
# --------------------------------------------------------------------------
def download_group(group: str, args: argparse.Namespace) -> bool:
    dest = DEST[group]
    cmd = [
        sys.executable, "push_artifacts_to_wandb.py",
        "--download", group,
        "--dest", dest,
        "--project", args.project,
        "--version", args.version,
        "--strict",
    ]
    if args.entity:
        cmd += ["--entity", args.entity]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    print(f"[bootstrap] restoring W&B artifact group '{group}' -> {dest}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[bootstrap] ⚠ could not restore '{group}' (see message above)")
        return False
    return True


def verify_checkpoints() -> bool:
    ckpt = Path("codeswitch_ckpt")
    required = ("draft_cb0.pt", "residual_cb1_8.pt", "meta.json")
    missing = [name for name in required if not (ckpt / name).is_file()]
    if missing:
        print(f"[bootstrap] ⚠ checkpoints missing after restore: {missing}")
        print("[bootstrap]   The 'student' artifact is written by the train cell's "
              "`--group student` push.")
        print("[bootstrap]   If that push did not run (or the WANDB_API_KEY secret was "
              "absent), the .pt weights are not in the bundle zip either "
              "(the zip excludes codeswitch_ckpt/*.pt).")
        return False
    print("[bootstrap] checkpoints:")
    for name in required:
        p = ckpt / name
        print(f"             {name:20s} {p.stat().st_size / 1e6:8.2f} MB")
    return True


def teacher_shard_count() -> int:
    return len(list(Path("teacher_cache").glob("shard_*.npz")))


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
def unzip_bundle(dest: Path) -> None:
    zpath = Path("/kaggle/working/codeswitch_artifacts_2k.zip")
    if not zpath.is_file():
        print(f"[bootstrap] no bundle zip at {zpath}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap] unzipping {zpath} -> {dest}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)
    print(f"[bootstrap] bundle contents: {sorted(p.name for p in dest.iterdir())[:12]}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore deps + W&B artifacts, then run diagnose_audio_bug.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default=None,
                        help="Repo directory (default: /kaggle/working/codeSwitch)")
    parser.add_argument("--groups", default="student,teacher",
                        help="Comma-separated artifact groups: student, teacher, bundle")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--version", default="latest")
    parser.add_argument("--api-key", default=None,
                        help="Defaults to $WANDB_API_KEY or the Kaggle secret")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-diagnose", action="store_true")
    parser.add_argument("--diagnostic", default="diagnose_audio_bug.py")
    parser.add_argument("--diag-args", default="",
                        help="Extra arguments passed to the diagnostic (default: its own defaults)")
    parser.add_argument("--unzip-bundle", action="store_true")
    parser.add_argument("--unzip-dir", default="/kaggle/working/artifacts_2k")
    args = parser.parse_args(argv)

    repo = find_repo(args.repo)
    if repo is None:
        if args.repo:
            raise SystemExit(
                f"--repo {args.repo} does not contain run_full_evaluation.py; "
                "point it at the codeSwitch checkout")
        repo = clone_repo(DEFAULT_REPO)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    print(f"[bootstrap] repo: {repo}")

    # ---- 1. dependencies ------------------------------------------------
    if not args.skip_install:
        problems = dependency_problems()
        if problems:
            print("[bootstrap] missing/incompatible: " + "; ".join(problems))
            install_dependencies()
            problems = dependency_problems()
            if problems:
                print("[bootstrap] still missing after install: " + "; ".join(problems))
                return 1
        else:
            print("[bootstrap] dependencies already OK")
        # Import in a fresh process: this is the real check (protobuf/transformers
        # can be present but unimportable after a mixed install).
        if not test_import("parler_tts"):
            print("[bootstrap] parler_tts is installed but cannot be imported; "
                  "retrying the staged install once...")
            install_dependencies()
            if not test_import("parler_tts"):
                print("[bootstrap] parler_tts still cannot be imported.")
                print("[bootstrap] Try a fresh session, or: "
                      "pip install -q 'protobuf==5.29.5' && restart the kernel.")
                return 1
    else:
        print("[bootstrap] --skip-install: not checking dependencies")

    # ---- 2. W&B artifacts ----------------------------------------------
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if not args.skip_download:
        for group in groups:
            if group not in DEST:
                print(f"[bootstrap] unknown group '{group}'; expected {sorted(DEST)}")
                return 2
            download_group(group, args)
    else:
        print("[bootstrap] --skip-download: using local files")

    if not verify_checkpoints():
        print("[bootstrap] cannot run the full diagnostic without the student "
              "checkpoints.")
        if "bundle" in groups and args.unzip_bundle:
            unzip_bundle(Path(args.unzip_dir))
        return 1

    n_shards = teacher_shard_count()
    print(f"[bootstrap] teacher shards restored: {n_shards}")
    if n_shards == 0:
        print("[bootstrap] ⚠ no teacher_cache shards: the diagnostic will skip the "
              "cache/shard sections but still compare teacher-vs-student on 2 utterances.")

    if "bundle" in groups and args.unzip_bundle:
        unzip_bundle(Path(args.unzip_dir))

    # ---- 3. diagnostic (fresh process) ---------------------------------
    if args.skip_diagnose:
        print("[bootstrap] --skip-diagnose: stopping after restore")
        return 0

    diag = Path(args.diagnostic)
    if not diag.is_file():
        print(f"[bootstrap] ⚠ {diag} not found in {repo}.")
        print("[bootstrap]   Save the diagnostic script next to this file, or pass "
              "--diagnostic /path/to/diagnose_audio_bug.py")
        return 2

    # Run with the diagnostic's own defaults (the script already defaults to
    # codeswitch_ckpt / teacher_cache / /kaggle/working/audio_diag).  Extra
    # flags can be forwarded verbatim with --diag-args.
    cmd = [sys.executable, str(diag), *shlex.split(args.diag_args)]
    print("[bootstrap] running: " + " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
