"""Tests for push_to_wandb.py - all offline, wandb is faked.

The pusher is the only network-touching part of the suite, so it is written to
be importable and testable without ``wandb`` installed: the real package is
imported lazily inside :func:`push_to_wandb.push`, and these tests inject a
recording stub into ``sys.modules``.
"""
from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path

import pytest

import push_to_wandb as W


# ===========================================================================
# fixtures
# ===========================================================================
def _stats(vals, **extra):
    vals = sorted(float(v) for v in vals)
    out = {"n": len(vals), "mean": sum(vals) / len(vals), "p50": vals[len(vals) // 2],
           "p90": vals[min(len(vals) - 1, int(0.9 * len(vals)))]}
    out.update(extra)
    return out


def _tiny_wav(path: Path, seconds: float = 0.02, sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return path


def _summary_fixture() -> dict:
    def config_block(cfg, base):
        return {
            "n": 3, "audio_sec": _stats([3.0, 3.2, 2.8]),
            "ttfa_compute_warm": _stats([base, base + 2, base + 4]),
            "ttfa_e2e_warm": _stats([base + 1, base + 3, base + 5]),
            "ttfa_compute_cold": _stats([base + 100]),
            "ttfa_e2e_cold": _stats([base + 120]),
            "total": _stats([base * 20, base * 22, base * 24]),
            "rtf": _stats([base / 200.0]),
            "cer": _stats([0.2, 0.5, 0.9]),
            "cer_latin": _stats([0.1, 0.3, 0.6]),
            "wer": _stats([0.4, 0.6, 0.8]),
            "utmos": _stats([1.4, 1.5, 1.6]),
            "mcd_vs_a": _stats([2.0, 2.1, 2.2]) if cfg == "C" else {"n": 0, "mean": None},
            "accept_true": _stats([39.0, 40.0, 41.0]) if cfg == "C" else {"n": 0, "mean": None},
            "accept_synth": _stats([26.0, 27.0, 28.0]) if cfg == "C" else {"n": 0, "mean": None},
            "logq_draft": _stats([-2.9]) if cfg == "C" else {"n": 0, "mean": None},
            "logq_kept": _stats([-2.7]) if cfg == "C" else {"n": 0, "mean": None},
            "fallback_rate": 0.0, "fallback_utts": 0, "asr_fail_utts": 0,
            "registers": {name: {"n": 1, "cer": _stats([0.3]), "cer_latin": _stats([0.2]),
                                 "utmos": _stats([1.5])}
                          for name in ("native_tamil", "romanized", "transit")},
        }

    return {
        "n_rows": 9,
        "configs": ["A", "B", "C"],
        "aggregates": {
            "_session_cold": {"config": "A", "utt_id": "ta-01",
                              "ttfa_compute_ms": 1054.0, "ttfa_e2e_ms": 1056.0},
            "A": config_block("A", 320.0),
            "B": config_block("B", 32.0),
            "C": config_block("C", 70.0),
        },
        "gate_sweep": [
            {"threshold": 0.0, "n": 3, "fallback_utts": 0, "fallback_rate": 0.0,
             "ttfa_compute": _stats([70.0]), "total": _stats([210.0]), "rtf": _stats([0.07]),
             "cer": _stats([0.9]), "cer_latin": _stats([0.6]), "utmos": _stats([1.5]),
             "accept_true": _stats([40.0])},
            {"threshold": 0.3, "n": 3, "fallback_utts": 2, "fallback_rate": 0.5,
             "ttfa_compute": _stats([200.0]), "total": _stats([4000.0]), "rtf": _stats([1.3]),
             "cer": _stats([0.4]), "cer_latin": _stats([0.25]), "utmos": _stats([1.8]),
             "accept_true": _stats([41.0])},
        ],
        "gate_mode": "derived",
        "gate_thresholds": [0.0, 0.3],
        "gate_stat": "utt_min",
        "gate_validation": [{"threshold": 0.3, "fallback_rate": 0.5,
                             "total_p50_synth_ms": 4050.0, "total_p50_derived_ms": 4000.0,
                             "cer_p50_synth": 0.41, "cer_p50_derived": 0.4}],
        "gate_note": None,
        "switch_distance": [
            {"bucket": "0-1", "n_frames": 100, "n_utts": 3, "mean": 42.0, "std": 5.0,
             "n_paired": 3, "delta_vs_far": 1.5, "delta_ci_lo": -4.0, "delta_ci_hi": 7.0,
             "p_vs_far": 0.42},
            {"bucket": "far", "n_frames": 80, "n_utts": 3, "mean": 40.5, "std": 4.0,
             "n_paired": 3, "delta_vs_far": 0.0, "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
             "p_vs_far": 1.0},
        ],
        "figures": {},
        "tables_tex": "tables.tex",
    }


def _workspace(tmp_path: Path) -> dict:
    figures = tmp_path / "paper_figures"
    results_dir = tmp_path / "eval_results"
    audio_dir = tmp_path / "eval_audio"
    ckpt_dir = tmp_path / "codeswitch_ckpt"
    figures.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary = _summary_fixture()
    png = figures / "fig1_latency_quality_tradeoff.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    pdf = figures / "fig1_latency_quality_tradeoff.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    summary["figures"] = {"fig1": [str(png), str(pdf)]}
    (figures / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (figures / "tables.tex").write_text("% tables\n", encoding="utf-8")

    rows = []
    cer_lat = {"A": 0.62, "B": 0.94, "C": 0.99}
    for cfg in ("A", "B", "C"):
        for index, value in enumerate((0.10, 0.35, 0.80)):
            wav = _tiny_wav(audio_dir / cfg / f"ut-{index + 1:02d}.wav")
            rows.append({
                "config": cfg, "utt_id": f"ut-{index + 1:02d}",
                "register": ("native_tamil", "romanized", "transit")[index],
                "text": "test sentence", "gate_threshold": 0.0, "fallback": False,
                "error": None, "cer": value, "cer_latin": value * cer_lat[cfg],
                "audio_path": str(wav),
            })
    results = results_dir / "raw_results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (results_dir / "run_meta.json").write_text(json.dumps({
        "args": {"ckpt": "codeswitch_ckpt", "seed": 42},
        "n_utterances": 3, "device": "cuda:0", "dtype": "fp16",
        "checkpoint_dir": "codeswitch_ckpt", "draft_trained": True,
        "residual_trained": True, "scorer_available": True, "utmos_available": False,
        "utmos_model": "torch.hub:tarepan/SpeechMOS:v1.2.0/utmos22_strong",
        "versions": {"torch": "2.4.0"},
    }), encoding="utf-8")
    (ckpt_dir / "meta.json").write_text(json.dumps({
        "corpus_size": 300, "corpus_source": "codeswitch_corpus.txt",
        "target_source": "lm", "residual_scheduled_sampling": 0.5,
        "draft": {"best_val": 5.1126, "best_epoch": 10, "top1_pct": 7.13, "top5_pct": 21.32},
        "residual": {"best_val": 6.6032, "top1_cb1_8_pct": 1.65},
        "length_model": {"a": 3.84, "b": 95.5, "mae_frames": 33.1},
    }), encoding="utf-8")
    return {"figures": figures, "results": results, "audio": audio_dir,
            "ckpt": ckpt_dir, "summary": summary}


# ===========================================================================
# unit tests
# ===========================================================================
def test_resolve_api_key_order(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "env-key")
    assert W.resolve_api_key("explicit", verbose=False) == "explicit"
    assert W.resolve_api_key(verbose=False) == "env-key"
    monkeypatch.delenv("WANDB_API_KEY")
    # no env, no kaggle_secrets installed here -> None (ambient credentials)
    assert W.resolve_api_key(verbose=False) is None


def test_sanitize_key():
    assert W.sanitize_key("ttfa_compute_warm") == "ttfa_compute_warm"
    assert W.sanitize_key("CER p50 (%)") == "CER_p50"
    assert W.sanitize_key("a//b") == "a_b"
    assert W.sanitize_key("") == "value"


def test_headline_metrics_are_flat_numbers(tmp_path):
    ws = _workspace(tmp_path)
    metrics = W.headline_metrics(ws["summary"],
                                json.loads((ws["results"].parent / "run_meta.json").read_text()),
                                json.loads((ws["ckpt"] / "meta.json").read_text()))
    assert metrics["latency/ttfa_compute_p50_ms/A"] == pytest.approx(322.0)
    assert metrics["latency/ttfa_compute_cold_ms/A"] == pytest.approx(420.0)
    assert metrics["acceptance/cb0_true_mean_pct/C"] == pytest.approx(40.0)
    assert metrics["acceptance/ar_fallback_rate/C"] == pytest.approx(0.0)
    assert metrics["gate/ar_fallback_pct/tau_0.30"] == pytest.approx(50.0)
    assert metrics["gate/total_p50_ms/tau_0.30"] == pytest.approx(4000.0)
    assert metrics["switch/accept_true_mean_pct/0-1"] == pytest.approx(42.0)
    assert metrics["register_cer_lat_p50/native_tamil/C"] == pytest.approx(0.2)
    assert metrics["session/cold_ttfa_compute_ms/A"] == pytest.approx(1054.0)
    assert metrics["train/draft/top1_pct"] == pytest.approx(7.13)
    assert metrics["train/residual/top1_cb1_8_pct"] == pytest.approx(1.65)
    assert metrics["train/corpus_size"] == pytest.approx(300.0)
    assert metrics["env/utmos_available"] == 0.0
    assert all(isinstance(v, float) for v in metrics.values())
    assert not any("nan" in k.lower() for k in metrics)
    # NaN / None never leak into the scalar payload
    assert "quality/mcd_vs_a_mean/A" not in metrics


def test_table_specs_shape(tmp_path):
    ws = _workspace(tmp_path)
    specs = dict(W.table_specs(ws["summary"]))
    assert list(specs) == ["configs", "gate_sweep", "switch_distance", "registers"]
    assert specs["configs"]["columns"][0] == "Config"
    rows = specs["configs"]["rows"]
    assert [r[0] for r in rows] == ["A", "B", "C"]
    assert rows[0][1] == pytest.approx(322.0)          # TTFA-c p50
    assert rows[2][1] == pytest.approx(72.0)           # config C
    assert len(specs["gate_sweep"]["rows"]) == 2
    assert specs["gate_sweep"]["rows"][1][1] == pytest.approx(50.0)   # fallback %
    assert specs["switch_distance"]["rows"][0][0] == "0-1"
    assert len(specs["registers"]["rows"]) == 9        # 3 configs x 3 registers


def test_collect_figures_keeps_pngs_only(tmp_path):
    ws = _workspace(tmp_path)
    figures = W.collect_figures(ws["summary"], ws["figures"])
    assert len(figures) == 1
    key, path, caption = figures[0]
    assert key == "fig1_latency_quality_tradeoff"
    assert path.suffix == ".png" and path.is_file()
    assert "Fig. 1" in caption


def test_audio_selection_picks_spread_and_skips_missing(tmp_path):
    ws = _workspace(tmp_path)
    rows = W.load_rows(ws["results"])
    picks = W.audio_selection(rows, ws["audio"], max_per_config=3)
    keys = [key for key, _, _ in picks]
    assert len(keys) == 9 and len(set(keys)) == 9
    capt = {key: caption for key, _, caption in picks}
    assert "CER_lat" in capt["A_ut-01"]
    assert "native_tamil" in capt["A_ut-01"]
    # a row whose audio vanished is dropped instead of breaking the push
    rows[0]["audio_path"] = str(tmp_path / "gone.wav")
    fewer = W.audio_selection(rows, ws["audio"], max_per_config=1)
    assert all(key != "A_ut-01" for key, _, _ in fewer)
    assert W.audio_selection(rows, ws["audio"], max_per_config=0) == []


def test_audio_selection_prefers_cer_spread(tmp_path):
    ws = _workspace(tmp_path)
    picks = W.audio_selection(W.load_rows(ws["results"]), ws["audio"], max_per_config=3)
    a_ids = sorted(key for key, _, _ in picks if key.startswith("A_"))
    assert a_ids == ["A_ut-01", "A_ut-02", "A_ut-03"]     # 3 of 3, best..worst


def test_dry_run_needs_no_wandb_and_plans_everything(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", None)      # importing it would now explode
    report = W.push(results=ws["results"], figures_dir=ws["figures"], audio_dir=ws["audio"],
                    ckpt_dir=ws["ckpt"], dry_run=True, max_audio=2)
    assert report["dry_run"] and report["n_metrics"] > 60
    assert report["n_figures"] == 1 and report["n_tables"] == 4 and report["n_audio"] == 6
    assert "codeswitch_ckpt" in report["run_name"] and "derived" in report["run_name"]
    assert any(p.endswith("summary.json") for p in report["artifact_files"])


def test_push_logs_metrics_figures_tables_audio_and_artifact(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    report = W.push(results=ws["results"], figures_dir=ws["figures"], audio_dir=ws["audio"],
                    ckpt_dir=ws["ckpt"], project="proj", entity="team",
                    run_name="my-run", tags=["kaggle", "t4"], api_key="secret",
                    max_audio=1, verbose=False)
    assert report["url"] == "https://wandb.ai/team/proj/runs/abc123"
    assert fake.logins == [("secret",)]
    assert fake.init_kwargs["project"] == "proj" and fake.init_kwargs["entity"] == "team"
    assert fake.init_kwargs["name"] == "my-run"
    assert fake.init_kwargs["tags"] == ["kaggle", "t4"]
    assert fake.init_kwargs["config"]["gate_mode"] == "derived"
    assert fake.init_kwargs["config"]["corpus_size"] == 300
    # scalars
    metric_payloads = [payload for payload in fake.logged if "latency/ttfa_compute_p50_ms/A" in payload]
    assert metric_payloads and metric_payloads[0]["latency/ttfa_compute_p50_ms/A"] == pytest.approx(322.0)
    # media + tables
    image_keys = [k for payload in fake.logged for k in payload if k.startswith("figure/")]
    table_keys = [k for payload in fake.logged for k in payload if k.startswith("table/")]
    audio_keys = [k for payload in fake.logged for k in payload if k.startswith("audio/")]
    assert image_keys == ["figure/fig1_latency_quality_tradeoff"]
    assert set(table_keys) == {"table/configs", "table/gate_sweep",
                              "table/switch_distance", "table/registers"}
    assert len(audio_keys) == 3 and all(k.startswith("audio/") for k in audio_keys)
    assert fake.tables["table/configs"]["columns"][0] == "Config"
    assert fake.images[0][1]["caption"].startswith("Fig. 1")
    # artifact bundle carries the JSONL, tables and directories
    artifact = fake.artifacts[0]
    assert artifact["name"].startswith("codeswitch-eval-")
    assert artifact["type"] == "evaluation"
    assert any(str(f).endswith("raw_results.jsonl") for f in artifact["files"])
    assert str(ws["figures"]) in artifact["dirs"] and str(ws["audio"]) in artifact["dirs"]
    assert fake.finished == 1


def test_push_reports_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.build_plan(tmp_path / "nope.jsonl", tmp_path / "no_figures", tmp_path / "no_audio",
                     tmp_path / "no_ckpt", verbose=False)


def test_cli_exit_codes(tmp_path, capsys):
    ws = _workspace(tmp_path)
    assert W.main(["--dry-run", "--results", str(ws["results"]),
                   "--figures-dir", str(ws["figures"]), "--audio-dir", str(ws["audio"]),
                   "--ckpt-dir", str(ws["ckpt"])]) == 0
    # a missing summary is a warning by default, an error with --strict
    missing = ["--results", str(ws["results"]), "--figures-dir", str(tmp_path / "empty"),
               "--audio-dir", str(ws["audio"]), "--ckpt-dir", str(ws["ckpt"])]
    assert W.main(missing) == 0
    assert W.main(["--strict", *missing]) == 1
    assert "push failed" in capsys.readouterr().out


def test_push_offline_sets_wandb_mode(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    W.push(results=ws["results"], figures_dir=ws["figures"], audio_dir=ws["audio"],
            ckpt_dir=ws["ckpt"], offline=True, max_audio=0, verbose=False)
    assert __import__("os").environ["WANDB_MODE"] == "offline"


# ===========================================================================
# fake wandb
# ===========================================================================
class _FakeWandb:
    """Records everything the real client would do, for offline assertions."""

    def __init__(self):
        self.logged = []
        self.tables = {}
        self.images = []
        self.audios = []
        self.artifacts = []
        self.logins = []
        self.finished = 0
        self.init_kwargs = {}
        outer = self

        class _Table:
            def __init__(self, columns=None, data=None, **kwargs):
                self.columns = list(columns or [])
                self.data = [list(r) for r in (data or [])]
                outer.last_table = self

        class _Image:
            def __init__(self, path, caption=None, **kwargs):
                outer.images.append((path, {"caption": caption}))

        class _Audio:
            def __init__(self, path, caption=None, **kwargs):
                outer.audios.append((path, caption))

        class _Artifact:
            def __init__(self, name=None, type=None, metadata=None, **kwargs):
                self.name = name
                self.type = type
                self.metadata = metadata or {}
                self.files = []
                self.dirs = []
                outer.artifacts.append({"name": name, "type": type,
                                        "files": self.files, "dirs": self.dirs})

            def add_file(self, path, **kwargs):
                self.files.append(path)

            def add_dir(self, path, **kwargs):
                self.dirs.append(path)

        class _Run:
            url = "https://wandb.ai/team/proj/runs/abc123"
            id = "abc123"

            def log(self, payload, **kwargs):
                payload = dict(payload)
                outer.logged.append(payload)
                for key, value in payload.items():
                    if key.startswith("table/"):
                        outer.tables[key] = {"columns": value.columns, "data": value.data}

            def log_artifact(self, artifact, **kwargs):
                pass

            def finish(self):
                outer.finished += 1

        module = types.ModuleType("wandb")
        module.Table = _Table
        module.Image = _Image
        module.Audio = _Audio
        module.Artifact = _Artifact
        module.plot = types.SimpleNamespace(line=lambda *a, **k: {"plot": "line"})

        def _login(key=None, **kwargs):
            outer.logins.append((key,))

        def _init(**kwargs):
            outer.init_kwargs = kwargs
            return _Run()

        module.login = _login
        module.init = _init
        self.module = module
