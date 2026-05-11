from __future__ import annotations

import io
import py_compile
from pathlib import Path

from pyflakes.api import check
from pyflakes.reporter import Reporter


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_PATH = REPO_ROOT / "agent.py"


def _pyflakes_output(path: Path) -> tuple[int, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    reporter = Reporter(stdout, stderr)
    warnings = check(path.read_text(encoding="utf-8"), str(path), reporter)
    output = stdout.getvalue() + stderr.getvalue()
    return warnings, output


def test_agent_py_exists() -> None:
    assert AGENT_PATH.is_file(), f"expected agent.py at {AGENT_PATH}"


def test_agent_py_compiles() -> None:
    py_compile.compile(str(AGENT_PATH), doraise=True)


def test_agent_py_has_no_pyflakes_findings() -> None:
    warnings, output = _pyflakes_output(AGENT_PATH)
    assert warnings == 0, output or "pyflakes reported warnings"
