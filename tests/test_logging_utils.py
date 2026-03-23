from pathlib import Path

from logging_utils import configure_logging


def test_configure_logging_writes_log_file(tmp_path: Path):
    logger = configure_logging(tmp_path)
    logger.info("hello tensorforge")

    log_path = tmp_path / "tensorforge.log"
    assert log_path.exists()
    assert "hello tensorforge" in log_path.read_text(encoding="utf-8")
