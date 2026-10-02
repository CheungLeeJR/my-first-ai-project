import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reproducibility import environment_metadata, file_sha256


def test_file_sha256_is_stable(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("research", encoding="utf-8")
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64


def test_environment_metadata_has_python_and_packages():
    meta = environment_metadata()
    assert meta["python"]
    assert "scikit-learn" in meta["packages"]
