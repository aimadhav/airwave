import io
from pathlib import Path
import tarfile
import zipfile
import stat
import pytest
import scripts.setup_diarization as setup


def test_asset_destinations_cannot_escape_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    assert setup.checked_target("models/foo.onnx") == tmp_path / "models/foo.onnx"
    for target in ["../outside", "app.py", "data/private/file", "models/../../outside"]:
        with pytest.raises(ValueError):
            setup.checked_target(target)


def test_verified_archive_extracts_only_allowlisted_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    archive = tmp_path / "models" / "test.tar.bz2"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:bz2") as target:
        for name, data in [("model/model.onnx", b"model-data"), ("model/do_not_run.py", b"bad-code")]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    asset = {"path": "models/test.tar.bz2", "sha256": setup.sha256(archive),
             "extract_members": ["model/model.onnx"]}
    setup.prepare_asset(asset, 1)
    assert (tmp_path / "models/model/model.onnx").read_bytes() == b"model-data"
    assert not (tmp_path / "models/model/do_not_run.py").exists()
    setup.prepare_asset(asset, 1)  # Idempotent, no redownload.
    (tmp_path / "models/model/model.onnx").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="differs"):
        setup.prepare_asset(asset, 1)


def test_verified_zip_extracts_only_allowlisted_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    archive = tmp_path / "models/runtime.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("runtime/mock.exe", b"not executable; fixture")
        writer.writestr("runtime/unrequested.txt", b"unused")
    asset = {"path": "models/runtime.zip", "sha256": setup.sha256(archive),
             "extract_zip_members": ["runtime/mock.exe"]}
    setup.prepare_asset(asset, 1)
    assert (tmp_path / "models/runtime/mock.exe").read_bytes() == b"not executable; fixture"
    assert not (tmp_path / "models/runtime/unrequested.txt").exists()
    setup.prepare_asset(asset, 1)
    (tmp_path / "models/runtime/mock.exe").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        setup.prepare_asset(asset, 1)


@pytest.mark.parametrize("name,symlink", [("../../outside", False), ("runtime/link", True)])
def test_zip_rejects_traversal_and_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, symlink: bool) -> None:
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    archive = tmp_path / "models/runtime.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as writer:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        writer.writestr(info, b"fixture")
    with pytest.raises(ValueError):
        setup.prepare_asset({"path": "models/runtime.zip", "sha256": setup.sha256(archive),
                             "extract_zip_members": [name]}, 1)
    assert not (tmp_path / "outside").exists()
