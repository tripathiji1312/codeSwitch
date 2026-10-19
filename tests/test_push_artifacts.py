"""Tests for push_artifacts_to_wandb.py - all offline, wandb is faked.

This is the disaster-recovery path for the 5.5 h teacher pass, so the parts that
must never lie are pinned here:

  * the *artifact layout contract* - the entry names inside each artifact are
    fixed (``teacher_cache/...``, ``codeswitch_corpus.txt`` at the root, ...),
    because ``--download ... --dest .`` guarantees the restore layout;
  * the collection/counting metadata (shard count, corpus lines, sizes);
  * the push wiring (one artifact per group, scalars, wait, report refs);
  * graceful degradation (missing paths warn instead of raising, an upload
    wait failure is tolerated);
  * the download reference format and the post-download verification counts.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

import push_artifacts_to_wandb as A


# ===========================================================================
# fixtures
# ===========================================================================
def _tree(root: Path, shards: int = 3, corpus_lines: int = 5) -> Path:
    """A miniature repo workspace with every group's files."""
    (root / "teacher_cache").mkdir(parents=True, exist_ok=True)
    for index in range(shards):
        (root / "teacher_cache" / f"shard_{index:04d}.npz").write_bytes(b"n" * (16 + index))
    (root / "codeswitch_corpus.txt").write_text(
        "\n".join(f"sentence {index}" for index in range(corpus_lines)) + "\n", encoding="utf-8")
    (root / "codeswitch_corpus.stats.json").write_text('{"n": 5}', encoding="utf-8")
    (root / "codeswitch_ckpt").mkdir(exist_ok=True)
    for name in ("draft_cb0.pt", "residual_cb1_8.pt", "meta.json"):
        (root / "codeswitch_ckpt" / name).write_bytes(b"m" * 32)
    (root / "codeswitch_artifacts_2k.zip").write_bytes(b"z" * 48)
    return root


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    monkeypatch.chdir(_tree(tmp_path))
    return tmp_path


# ===========================================================================
# collection: layout contract + counts
# ===========================================================================
def test_teacher_collect_counts_and_layout(tree):
    plan = A.collect_group("teacher", verbose=False)
    assert plan["artifact"] == "codeswitch-teacher-cache"
    assert plan["type"] == "dataset"
    assert plan["missing"] == [] and plan["empty"] is False
    entry_names = {name for _, name in plan["files"] + plan["dirs"]}
    assert entry_names == {"teacher_cache", "codeswitch_corpus.txt",
                           "codeswitch_corpus.stats.json"}
    # the cache is content-addressed: shard count + corpus size ride in metadata
    assert plan["metadata"]["shards"] == 3
    assert plan["metadata"]["corpus_lines"] == 5
    assert plan["metadata"]["corpus_txt"] == "codeswitch_corpus.txt"
    assert plan["n_files"] == 5 and plan["metadata"]["size_bytes"] > 0
    assert plan["metadata"]["recover"].endswith("--download teacher --dest .")
    json.dumps(plan["metadata"])  # wandb metadata must be JSON-serializable


def test_student_and_bundle_collect(tree):
    student = A.collect_group("student", verbose=False)
    assert student["dirs"] == [((Path.cwd() / "codeswitch_ckpt").resolve(), "codeswitch_ckpt")]
    assert student["metadata"]["ckpt_files"] == ["draft_cb0.pt", "meta.json",
                                                 "residual_cb1_8.pt"]
    assert student["n_files"] == 3

    bundle = A.collect_group("bundle", verbose=False)
    assert bundle["files"] == [((Path.cwd() / "codeswitch_artifacts_2k.zip").resolve(),
                                "codeswitch_artifacts_2k.zip")]
    assert bundle["metadata"]["recover"].endswith("--dest /kaggle/working")


def test_missing_paths_are_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing here
    plan = A.collect_group("teacher", verbose=False)
    assert plan["empty"] is True and plan["n_files"] == 0
    assert sorted(plan["missing"]) == ["codeswitch_corpus.stats.json",
                                       "codeswitch_corpus.txt", "teacher_cache"]
    report = A.push(["teacher"], dry_run=True, verbose=False)
    assert report["pushed"] == [] and report["groups"]["teacher"]["ref"] is None


def test_paths_override_changes_local_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    alt = tmp_path / "somewhere_else.zip"
    alt.write_bytes(b"z")
    plan = A.collect_group("bundle", paths={"codeswitch_artifacts_2k.zip": str(alt)},
                           verbose=False)
    assert plan["files"] == [(alt.resolve(), "codeswitch_artifacts_2k.zip")]  # artifact name is fixed
    assert plan["missing"] == []


def test_artifact_ref_formats():
    assert A.artifact_ref("teacher") == "codeswitch-tts/codeswitch-teacher-cache:latest"
    assert A.artifact_ref("student", entity="team", version="v2") == \
        "team/codeswitch-tts/codeswitch-student:v2"
    assert A.artifact_ref("bundle", project="proj", entity="me", version="v9") == \
        "me/proj/codeswitch-artifacts:v9"


def test_default_run_name_includes_corpus_lines(tree):
    plan = A.collect_group("teacher", verbose=False)
    name = A.default_run_name(["teacher"], [plan])
    assert name.startswith("backup-teacher-") and name.endswith("-c5")


# ===========================================================================
# dry run
# ===========================================================================
def test_dry_run_all_needs_no_wandb(tree, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)  # any import would now explode
    assert A.main(["--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    for artifact in ("codeswitch-teacher-cache", "codeswitch-student", "codeswitch-artifacts"):
        assert artifact in out
    assert "--download teacher --dest ." in out
    assert "shards: 3" in out


# ===========================================================================
# push wiring
# ===========================================================================
def test_push_logs_one_artifact_per_group(tree, monkeypatch):
    tree = Path.cwd()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    report = A.push(["teacher", "student"], project="proj", entity="team",
                    run_name="my-backup", tags=["kaggle", "backup"], api_key="secret",
                    wait_timeout=60.0, verbose=False)
    assert fake.logins == [("secret",)]
    assert fake.init_kwargs["project"] == "proj" and fake.init_kwargs["entity"] == "team"
    assert fake.init_kwargs["name"] == "my-backup"
    assert fake.init_kwargs["job_type"] == "backup"
    assert fake.init_kwargs["tags"] == ["kaggle", "backup"]
    assert fake.finished == 1

    teacher, student = fake.artifacts
    assert (teacher["name"], teacher["type"]) == ("codeswitch-teacher-cache", "dataset")
    assert (student["name"], student["type"]) == ("codeswitch-student", "model")
    assert teacher["metadata"]["shards"] == 3
    assert (str((tree / "teacher_cache").resolve()), "teacher_cache") in teacher["dirs"]
    assert (str((tree / "codeswitch_corpus.txt").resolve()),
            "codeswitch_corpus.txt") in teacher["files"]
    assert student["dirs"] == [(str((tree / "codeswitch_ckpt").resolve()), "codeswitch_ckpt")]
    assert fake.wait_calls == [60, 60]
    # scalars + refs
    payloads = [p for p in fake.logged if "backup/teacher/n_files" in p]
    assert payloads and payloads[0]["backup/teacher/n_files"] == 5
    assert report["groups"]["teacher"]["ref"] == "team/proj/codeswitch-teacher-cache:v3"
    assert report["groups"]["student"]["ref"] == "team/proj/codeswitch-student:v3"
    assert report["pushed"] == ["teacher", "student"]


def test_push_tolerates_wait_failure(tree, monkeypatch, capsys):
    fake = _FakeWandb()
    fake.wait_error = RuntimeError("offline mode")
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    report = A.push(["teacher"], wait_timeout=10.0)
    assert report["pushed"] == ["teacher"]
    assert report["groups"]["teacher"]["ref"] is None
    assert "upload-wait failed" in capsys.readouterr().out


def test_push_offline_sets_wandb_mode(tree, monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    A.push(["bundle"], offline=True, verbose=False)
    assert os.environ["WANDB_MODE"] == "offline"
    monkeypatch.delenv("WANDB_MODE", raising=False)


def test_push_skips_run_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    report = A.push(["teacher", "student", "bundle"], verbose=False)
    assert report["pushed"] == []
    assert fake.init_kwargs == {} and fake.artifacts == []  # no run was created


# ===========================================================================
# download / recovery
# ===========================================================================
def test_download_dry_run_ref(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)
    report = A.download("teacher", str(tmp_path), dry_run=True)
    assert report["ref"] == "codeswitch-tts/codeswitch-teacher-cache:latest"
    assert "downloading codeswitch-tts/codeswitch-teacher-cache:latest" in capsys.readouterr().out


def test_download_teacher_verifies_shards(tmp_path, monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    dest = tmp_path / "restore"
    (dest / "teacher_cache").mkdir(parents=True)
    for index in range(3):
        (dest / "teacher_cache" / f"shard_{index:04d}.npz").write_bytes(b"n")
    (dest / "codeswitch_corpus.txt").write_text("sentence 1\n", encoding="utf-8")
    report = A.download("teacher", str(dest), entity="team", version="v2")
    assert fake.artifact_refs == ["team/codeswitch-tts/codeswitch-teacher-cache:v2"]
    assert fake.download_roots == [str(dest)]
    assert report["shards"] == 3 and report["path"] == str(dest)


def test_download_student_and_bundle_verify(tmp_path, monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    dest = tmp_path / "restore"
    (dest / "codeswitch_ckpt").mkdir(parents=True)
    (dest / "codeswitch_ckpt" / "draft_cb0.pt").write_bytes(b"m")
    (dest / "codeswitch_ckpt" / "meta.json").write_text("{}", encoding="utf-8")
    (dest / "codeswitch_artifacts_2k.zip").write_bytes(b"z")
    student = A.download("student", str(dest), verbose=False)
    bundle = A.download("bundle", str(dest), verbose=False)
    assert student["ckpt_files"] == ["draft_cb0.pt", "meta.json"]
    assert bundle["zips"] == ["codeswitch_artifacts_2k.zip"]


# ===========================================================================
# CLI
# ===========================================================================
def test_cli_download_requires_dest():
    with pytest.raises(SystemExit) as excinfo:
        A.main(["--download", "teacher"])
    assert excinfo.value.code == 2


def test_cli_download_rejects_groups():
    with pytest.raises(SystemExit) as excinfo:
        A.main(["--download", "teacher", "--dest", ".", "--group", "student"])
    assert excinfo.value.code == 2


def test_cli_group_parsing_and_errors(tree):
    assert A.main(["--group", "teacher,student", "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        A.main(["--group", "bogus", "--dry-run"])
    with pytest.raises(SystemExit):
        A.main([])  # neither --group nor --all
    with pytest.raises(SystemExit):
        A.main(["--group", "teacher", "--download", "student", "--dest", "."])


# ===========================================================================
# fake wandb
# ===========================================================================
class _FakeWandb:
    """Records the artifact API calls; optionally makes ``wait`` fail."""

    def __init__(self):
        self.init_kwargs = {}
        self.logged = []
        self.logins = []
        self.finished = 0
        self.artifacts = []
        self.wait_calls = []
        self.wait_error = None
        self.artifact_refs = []
        self.download_roots = []
        self.api_keys = []
        outer = self

        class _Artifact:
            def __init__(self, name=None, type=None, metadata=None, **kwargs):
                self.name = name
                self.entry = {"name": name, "type": type, "metadata": metadata or {},
                              "files": [], "dirs": []}
                outer.artifacts.append(self.entry)

            def add_file(self, path, name=None, **kwargs):
                self.entry["files"].append((str(path), name or Path(path).name))

            def add_dir(self, path, name=None, **kwargs):
                self.entry["dirs"].append((str(path), name))

        class _Logged:
            def __init__(self, name):
                self._name = name

            def wait(self, timeout=None):
                outer.wait_calls.append(timeout)
                if outer.wait_error is not None:
                    raise outer.wait_error
                return self

            @property
            def qualified_name(self):
                if outer.wait_error is not None:
                    raise RuntimeError("not logged")
                return f"team/proj/{self._name}:v3"

        class _Run:
            url = "https://wandb.ai/team/proj/runs/abc"
            id = "abc"

            def log(self, payload, **kwargs):
                outer.logged.append(dict(payload))

            def log_artifact(self, artifact, **kwargs):
                return _Logged(artifact.name)

            def finish(self):
                outer.finished += 1

        class _ArtifactHandle:
            def __init__(self, ref):
                self.ref = ref

            def download(self, root=None, **kwargs):
                outer.download_roots.append(root)
                return str(root)

        class _Api:
            def __init__(self, api_key=None, **kwargs):
                outer.api_keys.append(api_key)

            def artifact(self, ref, type=None):
                outer.artifact_refs.append(ref)
                return _ArtifactHandle(ref)

        module = types.ModuleType("wandb")
        module.Artifact = _Artifact
        module.login = lambda key=None, **kw: outer.logins.append((key,))
        module.init = lambda **kw: (outer.init_kwargs.update(kw), _Run())[1]
        module.Api = _Api
        self.module = module
