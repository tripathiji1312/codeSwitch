#!/usr/bin/env python3
"""Push CodeSwitch-TTS *work products* to Weights & Biases as artifacts.

The expensive Kaggle steps do not survive a session death: the 2000-sentence
teacher pass is ~5.5 h and ``/kaggle/working`` is wiped with the session.  This
script snapshots each expensive output to W&B *right after it is produced*, so a
later session can pull it back with one command::

    # end of the teacher cell (~5.5 h of work):
    !python push_artifacts_to_wandb.py --group teacher
    # end of the draft/residual cell:
    !python push_artifacts_to_wandb.py --group student
    # after the final zip:
    !python push_artifacts_to_wandb.py --group bundle

Recovery (fresh session, from the repo directory)::

    !python push_artifacts_to_wandb.py --download teacher --dest .
    !python push_artifacts_to_wandb.py --download student --dest .
    !python push_artifacts_to_wandb.py --download bundle  --dest /kaggle/working

Artifacts keep the repo-relative layout (``teacher_cache/shard_*.npz``,
``codeswitch_corpus.txt`` at the root, ``codeswitch_ckpt/*.pt``, ...), so
``--dest .`` restores exactly what the trainer expects.  Teacher shards are
content-addressed per *sentence*, so a cache is only usable together with the
same corpus file - both ride in the same ``codeswitch-teacher-cache`` version.

Each push becomes a new version of the same artifact name (``:latest`` always
points at the newest).  The API key is resolved as ``--api-key`` →
``$WANDB_API_KEY`` → the Kaggle secret of the same name.  Failures warn and exit
0 unless ``--strict``, so a notebook cell never dies on a flaky upload;
``--dry-run`` prints the plan without importing ``wandb`` or touching the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from push_to_wandb import DEFAULT_PROJECT, SECRET_NAME, resolve_api_key

#: group → what it snapshots.  Entries are ``(kind, local_default, artifact_name)``;
#: the *artifact_name* is fixed so ``--download ... --dest .`` always restores the
#: same repo-relative layout, whatever local path the run happened to use.
GROUP_SPECS: Dict[str, Dict[str, Any]] = {
    "teacher": {
        "artifact": "codeswitch-teacher-cache",
        "type": "dataset",
        "entries": [
            ("dir", "teacher_cache", "teacher_cache"),
            ("file", "codeswitch_corpus.txt", "codeswitch_corpus.txt"),
            ("file", "codeswitch_corpus.stats.json", "codeswitch_corpus.stats.json"),
        ],
    },
    "student": {
        "artifact": "codeswitch-student",
        "type": "model",
        "entries": [("dir", "codeswitch_ckpt", "codeswitch_ckpt")],
    },
    "bundle": {
        "artifact": "codeswitch-artifacts",
        "type": "bundle",
        "entries": [
            ("file", "codeswitch_artifacts_2k.zip", "codeswitch_artifacts_2k.zip"),
        ],
    },
}

#: where each group is meant to be restored to
RESTORE_DEST = {"teacher": ".", "student": ".", "bundle": "/kaggle/working"}


# ===========================================================================
# collection (pure - no wandb, no network)
# ===========================================================================
def group_entries(group: str, paths: Optional[Dict[str, str]] = None,
                  ) -> List[Tuple[str, str, str]]:
    """``(kind, local_path, artifact_name)`` list for a group, with overrides."""
    overrides = paths or {}
    return [(kind, overrides.get(local, local), name)
            for (kind, local, name) in GROUP_SPECS[group]["entries"]]


def _dir_stats(directory: Path) -> Tuple[int, int]:
    """``(n_files, bytes)`` under *directory*."""
    count = 0
    total = 0
    for path in directory.rglob("*"):
        if path.is_file():
            count += 1
            try:
                total += path.stat().st_size
            except OSError:  # unreadable file must not kill the backup
                pass
    return count, total


def _count_lines(path: Optional[Path]) -> int:
    if path is None:
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def collect_group(group: str, paths: Optional[Dict[str, str]] = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """Everything one group's push needs, resolved against the local disk."""
    if group not in GROUP_SPECS:
        raise ValueError(f"unknown group {group!r}; expected one of {sorted(GROUP_SPECS)}")
    spec = GROUP_SPECS[group]
    files: List[Tuple[Path, str]] = []
    dirs: List[Tuple[Path, str]] = []
    missing: List[str] = []
    for kind, local, name in group_entries(group, paths):
        path = Path(local)
        if kind == "dir":
            if path.is_dir():
                dirs.append((path.resolve(), name))
            else:
                missing.append(local)
        elif path.is_file():
            files.append((path.resolve(), name))
        else:
            missing.append(local)

    n_files = len(files) + sum(_dir_stats(directory)[0] for directory, _ in dirs)
    size_bytes = sum(path.stat().st_size for path, _ in files if path.is_file())
    size_bytes += sum(_dir_stats(directory)[1] for directory, _ in dirs)

    metadata: Dict[str, Any] = {
        "group": group,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": int(n_files),
        "size_mb": round(size_bytes / 1e6, 3),
        "size_bytes": int(size_bytes),
        "recover": (f"python push_artifacts_to_wandb.py --download {group} "
                    f"--dest {RESTORE_DEST[group]}"),
    }
    if group == "teacher":
        shards = sum(len(list(directory.glob("shard_*.npz"))) for directory, _ in dirs)
        corpus = next((path for path, _ in files if path.suffix == ".txt"), None)
        metadata.update(
            shards=int(shards),
            corpus_lines=int(_count_lines(corpus)),
            corpus_txt=corpus.name if corpus else None,
        )
    if group == "student":
        metadata["ckpt_files"] = sorted(
            path.name for directory, _ in dirs
            for path in directory.rglob("*") if path.is_file()
        )

    plan: Dict[str, Any] = {
        "group": group,
        "artifact": spec["artifact"],
        "type": spec["type"],
        "files": files,
        "dirs": dirs,
        "missing": missing,
        "n_files": int(n_files),
        "size_mb": metadata["size_mb"],
        "metadata": metadata,
        "empty": n_files == 0,
    }
    if verbose:
        state = "nothing found" if plan["empty"] else f"{n_files} files, {plan['size_mb']:.1f} MB"
        extra = ""
        if group == "teacher" and not plan["empty"]:
            extra = f" | shards: {metadata['shards']} | corpus: {metadata['corpus_lines']} lines"
        if plan["missing"]:
            extra += f" | missing: {', '.join(plan['missing'])}"
        print(f"[W&B] {group}: {state}{extra}")
    return plan


# ===========================================================================
# push
# ===========================================================================
def artifact_ref(group: str, project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
                 version: str = "latest") -> str:
    """The ``[entity/]project/name:version`` reference used for downloads."""
    name = GROUP_SPECS[group]["artifact"]
    return f"{entity}/{project}/{name}:{version}" if entity else f"{project}/{name}:{version}"


def default_run_name(groups: Sequence[str], plans: Sequence[Dict[str, Any]]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M")
    name = "backup-" + "+".join(groups) + "-" + stamp
    for plan in plans:
        if plan["group"] == "teacher" and plan["metadata"].get("corpus_lines"):
            name += f"-c{int(plan['metadata']['corpus_lines'])}"
            break
    return name


def push(groups: Sequence[str], paths: Optional[Dict[str, str]] = None,
         project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
         run_name: Optional[str] = None, notes: Optional[str] = None,
         tags: Sequence[str] = (), api_key: Optional[str] = None,
         offline: bool = False, dry_run: bool = False,
         wait_timeout: float = 1800.0, verbose: bool = True) -> Dict[str, Any]:
    """Logs one artifact per group to one W&B run.  Returns a report dict."""
    plans = [collect_group(group, paths, verbose=verbose) for group in groups]
    to_push = [plan for plan in plans if not plan["empty"]]
    for plan in plans:
        if plan["empty"] and verbose:
            print(f"[W&B] ⚠ {plan['group']}: nothing to push"
                  + (f" (missing: {', '.join(plan['missing'])})" if plan["missing"] else ""))

    report: Dict[str, Any] = {
        "project": project,
        "entity": entity,
        "dry_run": bool(dry_run),
        "pushed": [plan["group"] for plan in to_push],
        "groups": {
            plan["group"]: {
                "artifact": plan["artifact"],
                "n_files": plan["n_files"],
                "size_mb": plan["size_mb"],
                "missing": plan["missing"],
                "ref": None,
            }
            for plan in plans
        },
    }

    if dry_run:
        for plan in to_push:
            print(f"[W&B] dry-run: would log {plan['artifact']} ({plan['type']}) to "
                  f"{entity or '<default entity>'}/{project}")
            for path, name in plan["files"]:
                print(f"        file {path} -> {name}")
            for path, name in plan["dirs"]:
                print(f"        dir  {path} -> {name}/")
            print(f"        meta {json.dumps(plan['metadata'], sort_keys=True)}")
            print(f"        recover {plan['metadata']['recover']}")
        return report

    if not to_push:
        if verbose:
            print("[W&B] nothing to push (no files found) - skipping the run")
        return report

    try:
        import wandb  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"wandb is not installed ({exc}); run `pip install -q wandb`") from exc

    if offline:
        os.environ["WANDB_MODE"] = "offline"
    key = resolve_api_key(api_key, verbose=verbose)
    if key:
        wandb.login(key=key, relogin=True)

    name = run_name or default_run_name([plan["group"] for plan in to_push], to_push)
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "job_type": "backup",
        "name": name,
        "notes": notes,
        "tags": [str(t) for t in tags] or None,
    }
    if entity:
        init_kwargs["entity"] = entity
    run = wandb.init(**init_kwargs)

    for plan in to_push:
        run.log({f"backup/{plan['group']}/n_files": plan["n_files"],
                 f"backup/{plan['group']}/size_mb": plan["size_mb"]})
        artifact = wandb.Artifact(name=plan["artifact"], type=plan["type"],
                                  metadata=plan["metadata"])
        for path, artifact_name in plan["files"]:
            artifact.add_file(str(path), name=artifact_name)
        for path, artifact_name in plan["dirs"]:
            artifact.add_dir(str(path), name=artifact_name)

        logged = run.log_artifact(artifact)
        try:
            logged.wait(timeout=int(wait_timeout) if wait_timeout and wait_timeout > 0 else None)
        except Exception as exc:  # never kill the notebook on a slow/stuck upload
            if verbose:
                print(f"[W&B] ⚠ {plan['group']}: upload-wait failed ({type(exc).__name__}: {exc}); "
                      "the upload may still finish in the background - keep the session alive")
        ref = None
        try:
            ref = logged.qualified_name
        except Exception:
            ref = None
        report["groups"][plan["group"]]["ref"] = ref
        if verbose:
            print(f"[W&B] {plan['group']} → {ref or plan['artifact']} "
                  f"({plan['n_files']} files, {plan['size_mb']:.1f} MB)")
            print(f"[W&B]   recover: {plan['metadata']['recover']}")

    url = getattr(run, "url", None)
    if url is None and hasattr(run, "get_url"):
        try:
            url = run.get_url()
        except Exception:
            url = None
    report["run_url"] = url
    report["run_name"] = name
    try:
        run.finish()
    except Exception:
        pass
    return report


# ===========================================================================
# recovery
# ===========================================================================
def download(group: str, dest: str, project: str = DEFAULT_PROJECT,
             entity: Optional[str] = None, version: str = "latest",
             api_key: Optional[str] = None, dry_run: bool = False,
             verbose: bool = True) -> Dict[str, Any]:
    """Restores one group's latest artifact into *dest* (repo-relative layout)."""
    ref = artifact_ref(group, project=project, entity=entity, version=version)
    report: Dict[str, Any] = {"group": group, "ref": ref, "dest": str(dest),
                              "dry_run": bool(dry_run)}
    if verbose:
        print(f"[W&B] downloading {ref} -> {dest}")
    if dry_run:
        return report

    try:
        import wandb  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"wandb is not installed ({exc}); run `pip install -q wandb`") from exc

    key = resolve_api_key(api_key, verbose=verbose)
    api = wandb.Api(api_key=key) if key else wandb.Api()
    artifact = api.artifact(ref)
    path = artifact.download(root=str(dest))
    report["path"] = str(path)

    dest_path = Path(dest)
    if group == "teacher":
        shards = len(list(dest_path.glob("teacher_cache/shard_*.npz")))
        report["shards"] = shards
        if verbose:
            print(f"[W&B] restored {shards} teacher shards -> {dest_path / 'teacher_cache'}")
            print("[W&B] shards are content-addressed per sentence: the restored "
                  "codeswitch_corpus.txt must stay byte-identical or the trainer re-collects")
    elif group == "student":
        ckpt_dir = dest_path / "codeswitch_ckpt"
        names = sorted(p.name for p in ckpt_dir.rglob("*") if p.is_file()) \
            if ckpt_dir.is_dir() else []
        report["ckpt_files"] = names
        if verbose:
            print(f"[W&B] restored checkpoints: {', '.join(names) or '(none found - check --dest)'}")
    else:
        zips = sorted(p.name for p in dest_path.glob("*.zip"))
        report["zips"] = zips
        if verbose:
            print(f"[W&B] restored zip(s): {', '.join(zips) or '(none found - check --dest)'}")
    return report


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Back up (or restore) CodeSwitch-TTS work products to/from W&B artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--group", action="append", default=None,
                        help="What to push: teacher | student | bundle "
                             "(repeatable or comma-separated).")
    parser.add_argument("--all", action="store_true", help="Push every group.")
    parser.add_argument("--download", choices=sorted(GROUP_SPECS), default=None,
                        help="Restore the latest artifact of this group instead of pushing.")
    parser.add_argument("--dest", default=None,
                        help="Restore destination ('.' rebuilds the repo layout). "
                             "Required with --download.")
    parser.add_argument("--version", default="latest",
                        help="Artifact version/alias to download.")
    parser.add_argument("--teacher-cache", default="teacher_cache")
    parser.add_argument("--corpus-txt", default="codeswitch_corpus.txt")
    parser.add_argument("--corpus-stats", default="codeswitch_corpus.stats.json")
    parser.add_argument("--ckpt-dir", default="codeswitch_ckpt")
    parser.add_argument("--zip", dest="zip_path", default="codeswitch_artifacts_2k.zip")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default=None,
                        help="Defaults to backup-<groups>-<timestamp>[-c<corpus lines>].")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", default="kaggle,backup", help="Comma-separated tags.")
    parser.add_argument("--api-key", default=None,
                        help=f"Defaults to ${SECRET_NAME}, then to the Kaggle secret of that name.")
    parser.add_argument("--offline", action="store_true",
                        help="WANDB_MODE=offline: write the run to disk instead of uploading.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded (no network, no wandb import).")
    parser.add_argument("--wait-timeout", type=float, default=1800.0,
                        help="Seconds to block until each artifact upload is committed "
                             "(0 = wait forever).")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when the push fails (default: warn and exit 0).")
    return parser


def _expand_groups(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    groups: List[str] = list(GROUP_SPECS) if args.all else []
    for chunk in args.group or []:
        for piece in str(chunk).split(","):
            piece = piece.strip()
            if piece:
                groups.append(piece)
    bad = [group for group in groups if group not in GROUP_SPECS]
    if bad:
        parser.error(f"unknown group(s) {bad}; expected {sorted(GROUP_SPECS)}")
    seen: List[str] = []
    for group in groups:  # de-dupe, keep order
        if group not in seen:
            seen.append(group)
    return seen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.download:
        if args.group or args.all:
            parser.error("--download cannot be combined with --group/--all")
        if not args.dest:
            parser.error("--dest is required with --download "
                         "(use '.' to restore the repo layout)")
        try:
            report = download(args.download, args.dest, project=args.project,
                              entity=args.entity, version=args.version,
                              api_key=args.api_key, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[W&B] ⚠ download failed: {type(exc).__name__}: {exc}")
            print(f"[W&B] reference was {artifact_ref(args.download, args.project, args.entity, args.version)}")
            print("[W&B] hints: the notebook's Internet switch must be on, and the key can be "
                  f"given as --api-key, ${SECRET_NAME} or the Kaggle secret '{SECRET_NAME}'.")
            return 1 if args.strict else 0
        if not args.dry_run:
            print(f"[W&B] done: {report['ref']} -> {report['dest']}")
        return 0

    groups = _expand_groups(args, parser)
    if not groups:
        parser.error("give --group teacher|student|bundle (repeatable) or --all")

    paths = {
        "teacher_cache": args.teacher_cache,
        "codeswitch_corpus.txt": args.corpus_txt,
        "codeswitch_corpus.stats.json": args.corpus_stats,
        "codeswitch_ckpt": args.ckpt_dir,
        "codeswitch_artifacts_2k.zip": args.zip_path,
    }
    try:
        report = push(
            groups, paths=paths, project=args.project, entity=args.entity,
            run_name=args.run_name, notes=args.notes,
            tags=[t.strip() for t in str(args.tags).split(",") if t.strip()],
            api_key=args.api_key, offline=args.offline, dry_run=args.dry_run,
            wait_timeout=args.wait_timeout,
        )
    except Exception as exc:
        print(f"[W&B] ⚠ backup failed: {type(exc).__name__}: {exc}")
        print("[W&B] hints: the notebook's Internet switch must be on, and the key can be "
              f"given as --api-key, ${SECRET_NAME} or the Kaggle secret '{SECRET_NAME}'.")
        return 1 if args.strict else 0
    if not args.dry_run and report.get("run_url"):
        print(f"[W&B] done: {report['run_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
