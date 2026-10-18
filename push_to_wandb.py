#!/usr/bin/env python3
"""Push a CodeSwitch-TTS evaluation run to Weights & Biases.

``run_full_evaluation.py`` already writes everything a reviewer needs into a few
files (``summary.json``, the LaTeX tables, the figures, the raw JSONL and the
audio).  This script turns that bundle into one W&B run:

* headline numbers → scalars grouped into panels (latency, quality, acceptance,
  target-distribution faithfulness, gate sweep, distance-to-switch, registers),
* the paper figures → ``wandb.Image`` (PNG, the PDFs ride along in the artifact),
* the sweep/table data → ``wandb.Table`` (readable in the UI, no pandas needed),
* a few *labelled* audio samples (best/median/worst CER per config),
* one artifact with the complete JSONL / figures / audio / ckpt meta for download.

Usage (Kaggle cell, after the evaluation cell):

    !python push_to_wandb.py --project codeswitch-tts --tags kaggle,t4

The API key is resolved as ``--api-key`` → ``$WANDB_API_KEY`` → the Kaggle secret
of the same name (Add-ons → Secrets).  ``--dry-run`` prints exactly what would be
uploaded without touching the network, and importing this module never requires
``wandb`` — it is imported only when a push actually happens, so the training and
evaluation scripts stay usable without it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PROJECT = "codeswitch-tts"
SECRET_NAME = "WANDB_API_KEY"

#: ``fig_paths`` keys in summary.json → panel titles
FIGURE_TITLES = {
    "fig1": "Fig. 1 - latency/quality trade-off",
    "fig2": "Fig. 2 - acceptance by zone",
    "fig3": "Fig. 3 - streaming timeline",
    "fig4": "Fig. 4 - scaling and registers",
    "fig5": "Fig. 5 - gate Pareto",
}


# ===========================================================================
# plumbing
# ===========================================================================
def resolve_api_key(explicit: Optional[str] = None, env_var: str = SECRET_NAME,
                    verbose: bool = True) -> Optional[str]:
    """Key from ``--api-key`` → environment → Kaggle secret (in that order)."""
    if explicit:
        return explicit
    if os.environ.get(env_var):
        if verbose:
            print(f"[W&B] using ${env_var} from the environment")
        return os.environ[env_var]
    try:  # only exists inside a Kaggle notebook
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret(env_var)
        if key:
            if verbose:
                print(f"[W&B] using the Kaggle secret '{env_var}'")
            return key
    except Exception:
        pass
    if verbose:
        print(f"[W&B] no key in --api-key/${env_var}/Kaggle secret; "
              "falling back to ambient W&B credentials (~/.netrc)")
    return None


def load_json(path: Any, warn: bool = True) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        if warn:
            print(f"[W&B] ⚠ could not read {path}: {type(exc).__name__}: {exc}")
        return None


def load_rows(path: Any) -> List[Dict[str, Any]]:
    """Reads the raw JSONL (blank/broken lines tolerated)."""
    rows: List[Dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        print(f"[W&B] ⚠ could not read {path}: {type(exc).__name__}: {exc}")
    return rows


def sanitize_key(text: Any) -> str:
    """W&B-safe key fragment."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "value"


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # NaN → absent
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _stat(stats: Any, field: str) -> Optional[float]:
    return _num((stats or {}).get(field)) if isinstance(stats, dict) else None


def _put(out: Dict[str, float], key: str, value: Any) -> None:
    number = _num(value)
    if number is not None:
        out[key] = number


def flatten_numeric(obj: Any, prefix: str, out: Dict[str, float],
                    depth: int = 0, limit: int = 400) -> None:
    """Collects numeric leaves of a nested dict into ``out`` (bounded)."""
    if len(out) >= limit or depth > 4:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_numeric(value, f"{prefix}/{sanitize_key(key)}", out, depth + 1, limit)
    elif isinstance(obj, (list, tuple)):
        return
    else:
        _put(out, prefix, obj)


# ===========================================================================
# metrics
# ===========================================================================
def headline_metrics(summary: Dict[str, Any],
                     run_meta: Optional[Dict[str, Any]] = None,
                     ckpt_meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Flat scalar dict for ``run.log`` (keys are pre-grouped by section)."""
    metrics: Dict[str, float] = {}
    aggregates = summary.get("aggregates") or {}

    for cfg, agg in aggregates.items():
        if not isinstance(agg, dict) or str(cfg).startswith("_"):
            continue
        base = sanitize_key(cfg)
        put = lambda key, value: _put(metrics, f"{key}/{base}", value)  # noqa: E731
        # ---- latency -------------------------------------------------------
        put("latency/ttfa_compute_p50_ms", _stat(agg.get("ttfa_compute_warm"), "p50"))
        put("latency/ttfa_compute_p90_ms", _stat(agg.get("ttfa_compute_warm"), "p90"))
        put("latency/ttfa_compute_mean_ms", _stat(agg.get("ttfa_compute_warm"), "mean"))
        put("latency/ttfa_e2e_p50_ms", _stat(agg.get("ttfa_e2e_warm"), "p50"))
        put("latency/ttfa_e2e_p90_ms", _stat(agg.get("ttfa_e2e_warm"), "p90"))
        put("latency/ttfa_compute_cold_ms", _stat(agg.get("ttfa_compute_cold"), "p50"))
        put("latency/ttfa_e2e_cold_ms", _stat(agg.get("ttfa_e2e_cold"), "p50"))
        put("latency/total_p50_ms", _stat(agg.get("total"), "p50"))
        put("latency/total_p90_ms", _stat(agg.get("total"), "p90"))
        put("latency/total_mean_ms", _stat(agg.get("total"), "mean"))
        put("latency/rtf_mean", _stat(agg.get("rtf"), "mean"))
        put("latency/rtf_p50", _stat(agg.get("rtf"), "p50"))
        # ---- quality -------------------------------------------------------
        put("quality/cer_p50", _stat(agg.get("cer"), "p50"))
        put("quality/cer_mean", _stat(agg.get("cer"), "mean"))
        put("quality/cer_latin_p50", _stat(agg.get("cer_latin"), "p50"))
        put("quality/cer_latin_mean", _stat(agg.get("cer_latin"), "mean"))
        put("quality/wer_p50", _stat(agg.get("wer"), "p50"))
        put("quality/utmos_mean", _stat(agg.get("utmos"), "mean"))
        put("quality/utmos_p50", _stat(agg.get("utmos"), "p50"))
        put("quality/mcd_vs_a_mean", _stat(agg.get("mcd_vs_a"), "mean"))
        put("quality/asr_fail_utts", agg.get("asr_fail_utts"))
        put("quality/audio_sec_mean", _stat(agg.get("audio_sec"), "mean"))
        # ---- acceptance / faithfulness -------------------------------------
        put("acceptance/cb0_true_mean_pct", _stat(agg.get("accept_true"), "mean"))
        put("acceptance/cb0_ref_mean_pct", _stat(agg.get("accept_synth"), "mean"))
        put("acceptance/ar_fallback_rate", agg.get("fallback_rate"))
        put("faithfulness/logq_draft_mean", _stat(agg.get("logq_draft"), "mean"))
        put("faithfulness/logq_kept_mean", _stat(agg.get("logq_kept"), "mean"))
        # ---- per-register quality (one panel, 3 registers x configs) --------
        for register, block in (agg.get("registers") or {}).items():
            if not isinstance(block, dict):
                continue
            reg = sanitize_key(register)
            _put(metrics, f"register_cer_lat_p50/{reg}/{base}", _stat(block.get("cer_latin"), "p50"))
            _put(metrics, f"register_cer_p50/{reg}/{base}", _stat(block.get("cer"), "p50"))
            _put(metrics, f"register_utmos_mean/{reg}/{base}", _stat(block.get("utmos"), "mean"))

    # ---- gate sweep (one operating point per threshold) --------------------
    for entry in summary.get("gate_sweep") or []:
        if not isinstance(entry, dict):
            continue
        tau = _num(entry.get("threshold"))
        tag = f"tau_{tau:.2f}" if tau is not None else "tau"
        _put(metrics, f"gate/ar_fallback_pct/{tag}", _num(entry.get("fallback_rate")) * 100.0
             if _num(entry.get("fallback_rate")) is not None else None)
        _put(metrics, f"gate/total_p50_ms/{tag}", _stat(entry.get("total"), "p50"))
        _put(metrics, f"gate/ttfa_compute_p50_ms/{tag}", _stat(entry.get("ttfa_compute"), "p50"))
        _put(metrics, f"gate/rtf_p50/{tag}", _stat(entry.get("rtf"), "p50"))
        _put(metrics, f"gate/cer_p50/{tag}", _stat(entry.get("cer"), "p50"))
        _put(metrics, f"gate/cer_lat_p50/{tag}", _stat(entry.get("cer_latin"), "p50"))
        _put(metrics, f"gate/utmos_mean/{tag}", _stat(entry.get("utmos"), "mean"))
        _put(metrics, f"gate/cb0_acc_mean_pct/{tag}", _stat(entry.get("accept_true"), "mean"))

    # ---- distance-to-switch (the code-switching-specific result) ----------
    for entry in summary.get("switch_distance") or []:
        if not isinstance(entry, dict):
            continue
        bucket = sanitize_key(entry.get("bucket"))
        _put(metrics, f"switch/accept_true_mean_pct/{bucket}", entry.get("mean"))
        _put(metrics, f"switch/delta_vs_far_pts/{bucket}", entry.get("delta_vs_far"))
        _put(metrics, f"switch/ci_lo/{bucket}", entry.get("delta_ci_lo"))
        _put(metrics, f"switch/ci_hi/{bucket}", entry.get("delta_ci_hi"))
        _put(metrics, f"switch/p_value/{bucket}", entry.get("p_vs_far"))
        _put(metrics, f"switch/n_frames/{bucket}", entry.get("n_frames"))

    # ---- session context ---------------------------------------------------
    cold = (aggregates.get("_session_cold") or {})
    if isinstance(cold, dict) and cold.get("config"):
        base = sanitize_key(cold["config"])
        _put(metrics, f"session/cold_ttfa_compute_ms/{base}", cold.get("ttfa_compute_ms"))
        _put(metrics, f"session/cold_ttfa_e2e_ms/{base}", cold.get("ttfa_e2e_ms"))
    _put(metrics, "session/n_eval_utterances", run_meta.get("n_utterances") if run_meta else None)
    _put(metrics, "session/n_rows", summary.get("n_rows"))

    # ---- training side (from the checkpoint meta) --------------------------
    if isinstance(ckpt_meta, dict):
        for group in ("draft", "residual", "length_model"):
            block = ckpt_meta.get(group)
            if isinstance(block, dict):
                flatten_numeric(block, f"train/{sanitize_key(group)}", metrics)
        if isinstance(ckpt_meta.get("versions"), dict):
            flatten_numeric(ckpt_meta["versions"], "train/versions", metrics)
        for key in ("corpus_size", "residual_scheduled_sampling"):
            _put(metrics, f"train/{sanitize_key(key)}", ckpt_meta.get(key))
    if isinstance(run_meta, dict):
        for key in ("n_utterances", "draft_trained", "residual_trained",
                    "scorer_available", "utmos_available"):
            value = run_meta.get(key)
            if isinstance(value, bool):
                metrics[f"env/{key}"] = float(value)
    return metrics


# ===========================================================================
# tables / figures / audio
# ===========================================================================
def _f2(value: Any, digits: int = 2) -> Any:
    number = _num(value)
    return None if number is None else round(number, digits)


def table_specs(summary: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """``[(name, {"columns": [...], "rows": [[...]]})]`` - pure data, no wandb."""
    specs: List[Tuple[str, Dict[str, Any]]] = []
    aggregates = summary.get("aggregates") or {}

    rows = []
    for cfg, agg in aggregates.items():
        if not isinstance(agg, dict) or str(cfg).startswith("_"):
            continue
        rows.append([
            str(cfg),
            _f2(_stat(agg.get("ttfa_compute_warm"), "p50"), 1),
            _f2(_stat(agg.get("total"), "p50"), 1),
            _f2(_stat(agg.get("rtf"), "mean"), 3),
            _f2(_stat(agg.get("cer"), "p50"), 3),
            _f2(_stat(agg.get("cer_latin"), "p50"), 3),
            _f2(_stat(agg.get("utmos"), "mean"), 3),
            _f2(_stat(agg.get("mcd_vs_a"), "mean"), 3),
            _f2(_stat(agg.get("accept_true"), "mean"), 2),
            _f2(_stat(agg.get("accept_synth"), "mean"), 2),
        ])
    specs.append(("configs", {
        "columns": ["Config", "TTFA-c p50 (ms)", "Total p50 (ms)", "RTF", "CER p50",
                    "CER_lat p50", "UTMOS", "MCD", "CB0 true (%)", "CB0 ref (%)"],
        "rows": rows,
    }))

    gate_rows = []
    for entry in summary.get("gate_sweep") or []:
        if not isinstance(entry, dict):
            continue
        rate = _num(entry.get("fallback_rate"))
        gate_rows.append([
            _f2(entry.get("threshold"), 4),
            None if rate is None else round(rate * 100.0, 1),
            _f2(_stat(entry.get("ttfa_compute"), "p50"), 1),
            _f2(_stat(entry.get("total"), "p50"), 1),
            _f2(_stat(entry.get("rtf"), "p50"), 3),
            _f2(_stat(entry.get("cer"), "p50"), 3),
            _f2(_stat(entry.get("cer_latin"), "p50"), 3),
            _f2(_stat(entry.get("utmos"), "mean"), 3),
            _f2(_stat(entry.get("accept_true"), "mean"), 2),
        ])
    specs.append(("gate_sweep", {
        "columns": ["tau", "AR fallback (%)", "TTFA-c p50 (ms)", "Total p50 (ms)", "RTF p50",
                    "CER p50", "CER_lat p50", "UTMOS", "CB0 acc (%)"],
        "rows": gate_rows,
    }))

    switch_rows = []
    for entry in summary.get("switch_distance") or []:
        if not isinstance(entry, dict):
            continue
        switch_rows.append([
            str(entry.get("bucket")),
            _num(entry.get("n_frames")),
            _f2(entry.get("mean"), 2),
            _f2(entry.get("delta_vs_far"), 2),
            f"[{_f2(entry.get('delta_ci_lo'), 1)}, {_f2(entry.get('delta_ci_hi'), 1)}]",
            _f2(entry.get("p_vs_far"), 3),
        ])
    specs.append(("switch_distance", {
        "columns": ["Distance to switch", "Frames", "Accept true (%)",
                    "delta vs far (pts)", "95% CI", "p"],
        "rows": switch_rows,
    }))

    register_rows = []
    for cfg, agg in aggregates.items():
        if not isinstance(agg, dict) or str(cfg).startswith("_"):
            continue
        for register, block in (agg.get("registers") or {}).items():
            if not isinstance(block, dict):
                continue
            register_rows.append([
                str(cfg), str(register), _num(block.get("n")),
                _f2(_stat(block.get("cer"), "p50"), 3),
                _f2(_stat(block.get("cer_latin"), "p50"), 3),
                _f2(_stat(block.get("utmos"), "mean"), 3),
            ])
    specs.append(("registers", {
        "columns": ["Config", "Register", "n", "CER p50", "CER_lat p50", "UTMOS"],
        "rows": register_rows,
    }))
    return [(name, spec) for name, spec in specs if spec["rows"]]


def collect_figures(summary: Dict[str, Any], figures_dir: Any,
                    ) -> List[Tuple[str, Path, str]]:
    """PNGs to embed as ``wandb.Image``: ``(key, path, caption)``.

    PDFs/TeX are not images - they ride along in the artifact bundle instead.
    """
    figures_dir = Path(figures_dir)
    out: List[Tuple[str, Path, str]] = []
    seen: set = set()

    def add(path: Path, caption: str) -> None:
        if path.suffix.lower() == ".png" and path.is_file() and str(path) not in seen:
            seen.add(str(path))
            out.append((sanitize_key(path.stem), path, caption))

    declared = summary.get("figures") or {}
    for name, paths in declared.items():
        if isinstance(paths, str):
            paths = [paths]
        caption = FIGURE_TITLES.get(str(name), str(name))
        for path in paths or []:
            add(Path(path), caption)
    if not out:  # JSONL from an older suite: fall back to whatever is on disk
        for path in sorted(figures_dir.glob("*.png")):
            add(path, path.stem)
    return out


def audio_selection(rows: Sequence[Dict[str, Any]], audio_dir: Any = None,
                    audio_dir_by_cfg: Any = None, max_per_config: int = 4,
                    ) -> List[Tuple[str, Path, str]]:
    """Best/median/worst-CER sample per config, captioned with its numbers."""
    if max_per_config <= 0:
        return []
    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or row.get("fallback"):
            continue
        if float(row.get("gate_threshold") or 0.0) > 0.0:
            continue
        cfg = str(row.get("config") or "?")
        by_cfg.setdefault(cfg, []).append(row)

    def quality(row: Dict[str, Any]) -> Tuple[int, float]:
        value = _num(row.get("cer_latin"))
        if value is None:
            value = _num(row.get("cer"))
        return (1, 0.0) if value is None else (0, float(value))

    out: List[Tuple[str, Path, str]] = []
    for cfg, cfg_rows in by_cfg.items():
        ordered = sorted(cfg_rows, key=quality)
        count = len(ordered)
        picks: List[Dict[str, Any]] = []
        wanted = [0, count // 2, count - 1]
        wanted += [int(round(i * (count - 1) / max(max_per_config - 1, 1)))
                   for i in range(max_per_config)]
        for index in wanted:
            if 0 <= index < count and len(picks) < max_per_config:
                if ordered[index] not in picks:
                    picks.append(ordered[index])
        for row in picks:
            path = row.get("audio_path")
            path = Path(path) if path else None
            if path is None or not path.is_file():
                fallback_dir = Path(audio_dir or "eval_audio")
                if audio_dir_by_cfg:
                    fallback_dir = fallback_dir / str(cfg)
                path = fallback_dir / f"{row.get('utt_id')}.wav"
            if not path.is_file():
                continue
            bits = [str(cfg), str(row.get("utt_id")), str(row.get("register") or "?")]
            cer = _num(row.get("cer"))
            cer_lat = _num(row.get("cer_latin"))
            if cer is not None:
                bits.append(f"CER {cer:.1%}")
            if cer_lat is not None:
                bits.append(f"CER_lat {cer_lat:.1%}")
            text = str(row.get("text") or "")
            if text:
                bits.append(text[:48])
            out.append((sanitize_key(f"{cfg}_{row.get('utt_id')}"), path, " | ".join(bits)))
    return out


# ===========================================================================
# the push
# ===========================================================================
def default_run_name(ckpt_dir: Any, ckpt_meta: Optional[Dict[str, Any]],
                     summary: Optional[Dict[str, Any]] = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M")
    name = f"eval-{stamp}-{Path(ckpt_dir).name}"
    corpus = _num((ckpt_meta or {}).get("corpus_size"))
    if corpus:
        name += f"-c{int(corpus)}"
    mode = (summary or {}).get("gate_mode")
    if mode:
        name += f"-{sanitize_key(mode)}"
    return name


def wandb_config(summary: Dict[str, Any], run_meta: Optional[Dict[str, Any]],
                 ckpt_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "configs": summary.get("configs"),
        "gate_mode": summary.get("gate_mode"),
        "gate_stat": summary.get("gate_stat"),
        "gate_thresholds": summary.get("gate_thresholds"),
        "n_rows": summary.get("n_rows"),
    }
    if isinstance(run_meta, dict):
        config["n_utterances"] = run_meta.get("n_utterances")
        config["device"] = run_meta.get("device")
        config["dtype"] = run_meta.get("dtype")
        config["checkpoint_dir"] = run_meta.get("checkpoint_dir")
        config["utmos_model"] = run_meta.get("utmos_model")
        config["eval_args"] = (run_meta.get("args") or {})
        config["versions"] = (run_meta.get("versions") or {})
    if isinstance(ckpt_meta, dict):
        config["corpus_size"] = ckpt_meta.get("corpus_size")
        config["corpus_source"] = ckpt_meta.get("corpus_source")
        config["target_source"] = ckpt_meta.get("target_source")
        config["residual_scheduled_sampling"] = ckpt_meta.get("residual_scheduled_sampling")
    return config


def build_plan(results: Any, figures_dir: Any, audio_dir: Any, ckpt_dir: Any,
               max_audio: int = 4, upload_audio: bool = True,
               upload_artifact: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Everything the push needs - collected without importing wandb."""
    results = Path(results)
    figures_dir = Path(figures_dir)
    ckpt_dir = Path(ckpt_dir)
    summary = load_json(figures_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(
            f"{figures_dir / 'summary.json'} not found - run run_full_evaluation.py first "
            "(it writes summary.json next to the figures)")
    run_meta = (load_json(results.parent / "run_meta.json")
                or load_json(figures_dir / "run_meta.json", warn=False))
    ckpt_meta = load_json(ckpt_dir / "meta.json",
                          warn=(ckpt_dir / "meta.json").exists())
    rows = load_rows(results)
    metrics = headline_metrics(summary, run_meta, ckpt_meta)
    tables = table_specs(summary)
    figures = collect_figures(summary, figures_dir)
    audio = audio_selection(rows, audio_dir, max_per_config=max_audio) if upload_audio else []
    artifacts: List[Path] = [path for path in [
        figures_dir / "summary.json", figures_dir / "tables.tex",
        results, results.parent / "run_meta.json", ckpt_dir / "meta.json",
    ] if path.is_file()]
    dirs = [path for path in ([figures_dir, ckpt_dir, audio_dir] if upload_artifact else [])
            if path is not None and Path(path).is_dir()]
    plan = {
        "summary": summary,
        "run_meta": run_meta,
        "ckpt_meta": ckpt_meta,
        "rows": rows,
        "metrics": metrics,
        "tables": tables,
        "figures": figures,
        "audio": audio,
        "artifact_files": artifacts,
        "artifact_dirs": dirs,
        "results": results,
        "figures_dir": figures_dir,
        "audio_dir": Path(audio_dir),
        "ckpt_dir": ckpt_dir,
    }
    if verbose:
        print(f"[W&B] plan: {len(metrics)} scalars | {len(figures)} figures | "
              f"{len(tables)} tables | {len(audio)} audio samples | "
              f"{len(artifacts) + len(dirs)} artifact entries")
    return plan


def push(results: Any = "eval_results/raw_results.jsonl", figures_dir: Any = "paper_figures",
         audio_dir: Any = "eval_audio", ckpt_dir: Any = "codeswitch_ckpt",
         project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
         run_name: Optional[str] = None, run_id: Optional[str] = None,
         group: Optional[str] = None, job_type: str = "evaluation",
         tags: Sequence[str] = (), notes: Optional[str] = None,
         api_key: Optional[str] = None, max_audio: int = 4, upload_audio: bool = True,
         upload_artifact: bool = True, offline: bool = False, dry_run: bool = False,
         verbose: bool = True) -> Dict[str, Any]:
    """Logs one evaluation run.  Returns a report dict (``url`` when logged)."""
    plan = build_plan(results, figures_dir, audio_dir, ckpt_dir, max_audio=max_audio,
                      upload_audio=upload_audio, upload_artifact=upload_artifact,
                      verbose=verbose)
    summary, run_meta, ckpt_meta = plan["summary"], plan["run_meta"], plan["ckpt_meta"]
    name = run_name or default_run_name(ckpt_dir, ckpt_meta, summary)
    report: Dict[str, Any] = {
        "run_name": name,
        "project": project,
        "entity": entity,
        "n_metrics": len(plan["metrics"]),
        "n_figures": len(plan["figures"]),
        "n_tables": len(plan["tables"]),
        "n_audio": len(plan["audio"]),
        "artifact_files": [str(p) for p in plan["artifact_files"]],
        "dry_run": bool(dry_run),
    }
    if dry_run:
        if verbose:
            print(f"[W&B] dry-run: would create run '{name}' in {entity or '<default entity>'}/{project}")
            for key, path, caption in plan["figures"]:
                print(f"        figure  {key} <- {path}  [{caption}]")
            for tick, spec in plan["tables"]:
                print(f"        table   {tick} ({len(spec['rows'])} rows)")
            for key, path, caption in plan["audio"]:
                print(f"        audio   {key} <- {path}  [{caption[:60]}]")
            for path in plan["artifact_files"]:
                print(f"        file    {path}")
            for path in plan["artifact_dirs"]:
                print(f"        dir     {path}")
            preview = list(sorted(plan["metrics"].items()))[:8]
            for key, value in preview:
                print(f"        metric  {key} = {value:g}")
        return report

    try:
        import wandb  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"wandb is not installed ({exc}); run `pip install -q wandb`") from exc

    if offline:
        os.environ["WANDB_MODE"] = "offline"
    key = resolve_api_key(api_key, verbose=verbose)
    if key:
        wandb.login(key=key, relogin=True)

    init_kwargs: Dict[str, Any] = {
        "project": project,
        "job_type": job_type,
        "name": name,
        "config": wandb_config(summary, run_meta, ckpt_meta),
        "tags": [str(t) for t in tags] or None,
        "notes": notes,
    }
    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    if run_id:
        init_kwargs.update(id=run_id, resume="allow")
    run = wandb.init(**init_kwargs)

    run.log(dict(plan["metrics"]))
    for key, path, caption in plan["figures"]:
        run.log({f"figure/{key}": wandb.Image(str(path), caption=caption)})
    gate_table = None
    for tick, spec in plan["tables"]:
        table = wandb.Table(columns=list(spec["columns"]), data=[list(r) for r in spec["rows"]])
        run.log({f"table/{tick}": table})
        if tick == "gate_sweep":
            gate_table = table
    if gate_table is not None and hasattr(wandb, "plot"):  # nicer Pareto panel
        try:
            run.log({
                "gate/pareto_cer_lat": wandb.plot.line(
                    gate_table, "tau", "CER_lat p50",
                    title="Gate Pareto: CER_lat vs tau (fallback rate in the table)"),
                "gate/pareto_total": wandb.plot.line(
                    gate_table, "tau", "Total p50 (ms)",
                    title="Gate Pareto: end-to-end latency vs tau"),
            })
        except Exception as exc:  # pragma: no cover - plotting is a bonus
            if verbose:
                print(f"[W&B] ⚠ plot.line skipped: {type(exc).__name__}: {exc}")
    for key, path, caption in plan["audio"]:
        try:
            run.log({f"audio/{key}": wandb.Audio(str(path), caption=caption)})
        except Exception as exc:  # pragma: no cover - audio is a bonus
            if verbose:
                print(f"[W&B] ⚠ audio {key} skipped: {type(exc).__name__}: {exc}")

    if upload_artifact:
        artifact = wandb.Artifact(
            name=f"codeswitch-eval-{name}",
            type="evaluation",
            metadata={"run_name": name, "project": project,
                      "n_rows": summary.get("n_rows"), "gate_mode": summary.get("gate_mode")},
        )
        for path in plan["artifact_files"]:
            artifact.add_file(str(path))
        for path in plan["artifact_dirs"]:
            if Path(path).is_dir():
                artifact.add_dir(str(path))
        run.log_artifact(artifact)
        report["artifact"] = artifact.name if hasattr(artifact, "name") else None

    url = getattr(run, "url", None)
    if url is None and hasattr(run, "get_url"):
        try:
            url = run.get_url()
        except Exception:
            url = None
    report["url"] = url
    report["run_id"] = getattr(run, "id", run_id)
    try:
        run.finish()
    except Exception:
        pass
    if verbose:
        print(f"[W&B] logged {report['n_metrics']} scalars, {report['n_figures']} figures, "
              f"{report['n_tables']} tables, {report['n_audio']} audio samples"
              + (f" | artifact {report.get('artifact')}" if report.get("artifact") else ""))
        if url:
            print(f"[W&B] run: {url}")
    return report


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Push a CodeSwitch-TTS evaluation run to Weights & Biases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", type=str, default="eval_results/raw_results.jsonl",
                   help="Raw JSONL written by run_full_evaluation.py.")
    p.add_argument("--figures-dir", type=str, default="paper_figures",
                   help="Directory holding summary.json / tables.tex / the figures.")
    p.add_argument("--audio-dir", type=str, default="eval_audio")
    p.add_argument("--ckpt-dir", type=str, default="codeswitch_ckpt")
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None,
                   help="Defaults to eval-<timestamp>-<ckpt>-c<corpus size>-<gate mode>.")
    p.add_argument("--run-id", type=str, default=None,
                   help="W&B run id to write into (with resume=allow) - use it to overwrite.")
    p.add_argument("--group", type=str, default=None, help="e.g. the corpus size or the paper run.")
    p.add_argument("--job-type", type=str, default="evaluation")
    p.add_argument("--tags", type=str, default="codeswitch,kaggle,t4",
                   help="Comma-separated tags.")
    p.add_argument("--notes", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None,
                   help=f"Defaults to ${SECRET_NAME}, then to the Kaggle secret of that name.")
    p.add_argument("--max-audio", type=int, default=4,
                   help="Audio samples per config (best/median/worst CER).")
    p.add_argument("--no-audio", dest="upload_audio", action="store_false")
    p.add_argument("--no-artifact", dest="upload_artifact", action="store_false")
    p.add_argument("--offline", action="store_true",
                   help="WANDB_MODE=offline: write the run to disk instead of uploading.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded (no network, no wandb import).")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero when the push fails (default: warn and exit 0).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = push(
            results=args.results, figures_dir=args.figures_dir, audio_dir=args.audio_dir,
            ckpt_dir=args.ckpt_dir, project=args.project, entity=args.entity,
            run_name=args.run_name, run_id=args.run_id, group=args.group,
            job_type=args.job_type,
            tags=[t.strip() for t in str(args.tags).split(",") if t.strip()],
            notes=args.notes, api_key=args.api_key, max_audio=args.max_audio,
            upload_audio=args.upload_audio, upload_artifact=args.upload_artifact,
            offline=args.offline, dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[W&B] ⚠ push failed: {type(exc).__name__}: {exc}")
        if args.project and "internet" not in str(exc).lower():
            print("[W&B] hints: the notebook's Internet switch must be on, and the key can be "
                  f"given as --api-key, ${SECRET_NAME} or the Kaggle secret '{SECRET_NAME}'.")
        return 1 if args.strict else 0
    if not args.dry_run and report.get("url"):
        print(f"[W&B] done: {report['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
