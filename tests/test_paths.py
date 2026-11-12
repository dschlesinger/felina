import importlib
from pathlib import Path

import data_tools.paths as paths


def _reload(monkeypatch, tmp_path, env_text=None, **env):
    for key in ("DATA_FOLDER", "FEATURE_FOLDER", "RESULTS_FOLDER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    if env_text is not None:
        (tmp_path / ".env").write_text(env_text)
    return importlib.reload(paths)


def test_defaults_without_env(monkeypatch, tmp_path):
    p = _reload(monkeypatch, tmp_path)
    assert (p.DATA_DIR, p.FEATURE_DIR, p.RESULTS_DIR) == (Path("data"), Path("data/features"), Path("results"))


def test_env_file_sets_folders(monkeypatch, tmp_path):
    p = _reload(monkeypatch, tmp_path, '# comment\nDATA_FOLDER=/d\nexport RESULTS_FOLDER="/r"\nFEATURE_FOLDER=\'/f\'\n')
    assert (p.DATA_DIR, p.FEATURE_DIR, p.RESULTS_DIR) == (Path("/d"), Path("/f"), Path("/r"))


def test_feature_folder_follows_data_folder(monkeypatch, tmp_path):
    p = _reload(monkeypatch, tmp_path, "DATA_FOLDER=/d\n")
    assert p.FEATURE_DIR == Path("/d/features")


def test_environment_wins_over_env_file(monkeypatch, tmp_path):
    p = _reload(monkeypatch, tmp_path, "DATA_FOLDER=/from_file\n", DATA_FOLDER="/from_env")
    assert p.DATA_DIR == Path("/from_env")
