"""D1 acceptance tests for dr_core.models.claude_cli.ChatClaudeCLI.

Per build-logs/task-claude-cli-wrapper.md and DECISIONS.md D1. T1-T3 invoke the
real `claude -p` CLI (a few cents; cold latency ~7s per call, per the task's
empirical contract) — no mocking, because the whole point is to validate the
real subprocess/accounting/cancellation/concurrency behavior. T4 is a pure
unit test (argv inspection only, no network).

Run: cd deer-flow/backend && uv run pytest packages/dr_core/ -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time

import pytest
from dr_core.models.claude_cli import ChatClaudeCLI


class _FakeProc:
    """Stand-in for asyncio.subprocess.Process, for tests that must not spawn a real
    `claude -p` process (M8a/M8b are argv/parsing-level fixes, not subprocess behavior)."""

    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 999999

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

# asyncio_mode = "auto" is set in packages/dr_core/pyproject.toml, so async
# test functions below need no explicit @pytest.mark.asyncio.


async def test_t1_accounting() -> None:
    """T1: ainvoke returns an AIMessage with populated usage_metadata and
    total_cost_usd in response_metadata."""
    model = ChatClaudeCLI(model="haiku")
    result = await model.ainvoke("Reply with exactly: OK")

    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0
    assert result.usage_metadata["output_tokens"] > 0
    assert result.usage_metadata["total_tokens"] == (result.usage_metadata["input_tokens"] + result.usage_metadata["output_tokens"])
    assert "input_token_details" in result.usage_metadata

    assert "total_cost_usd" in result.response_metadata
    assert result.response_metadata["total_cost_usd"] is not None


async def test_t2_cancellation_kills_subprocess() -> None:
    """T2 (kill-gate): cancelling/timing out an in-flight ainvoke must leave
    no orphaned `claude` process running. A cold claude -p call takes ~7s
    (per the task's empirical contract), so a 2s outer timeout reliably
    catches the process mid-flight without relying on a synthetic "slow
    prompt" that could itself finish before the timeout."""
    model = ChatClaudeCLI(model="haiku", request_timeout=600.0)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(model.ainvoke("Reply with exactly: OK"), timeout=2)

    proc = model._last_proc
    assert proc is not None, "wrapper never spawned a subprocess to kill"

    # Give the killpg() signal a brief moment to land, then confirm the
    # process is genuinely dead (not just abandoned/orphaned).
    for _ in range(20):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.25)

    assert proc.returncode is not None, "claude subprocess outlived the cancelled call (orphan)"

    # Cross-check via the OS: the pid must not be a live process anymore.
    import os
    import signal as _signal

    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)
    del _signal


async def test_t3_concurrency() -> None:
    """T3: N concurrent ainvoke calls all succeed and overlap (total
    wall-clock well under N times a single call), proving the event loop
    isn't blocked by subprocess IO."""
    model = ChatClaudeCLI(model="haiku")

    single_start = time.monotonic()
    single_result = await model.ainvoke("Reply with exactly: OK")
    single_elapsed = time.monotonic() - single_start
    assert single_result.content

    n = 3
    concurrent_start = time.monotonic()
    results = await asyncio.gather(*[model.ainvoke("Reply with exactly: OK") for _ in range(n)])
    concurrent_elapsed = time.monotonic() - concurrent_start

    assert len(results) == n
    for r in results:
        assert r.content

    assert concurrent_elapsed < n * single_elapsed, f"concurrent calls did not overlap: concurrent={concurrent_elapsed:.2f}s vs {n}x single={n * single_elapsed:.2f}s"


async def test_t5_agenerate_passes_a_neutral_cwd(monkeypatch) -> None:
    """M8a: claude -p must never run in the project tree (reloads CLAUDE.md/hooks per
    call). Asserts the cwd kwarg reaches create_subprocess_exec, without spawning a
    real process."""
    captured: dict = {}

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured.update(kwargs)
        payload = json.dumps({"result": "OK", "usage": {}, "total_cost_usd": 0.0, "is_error": False, "subtype": "success", "session_id": "s1"}).encode()
        return _FakeProc(stdout=payload)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    model = ChatClaudeCLI(model="haiku")
    result = await model.ainvoke("hi")

    assert result.content == "OK"
    assert captured.get("cwd") == tempfile.gettempdir()


async def test_t6_malformed_stdout_raises_descriptive_runtime_error(monkeypatch) -> None:
    """M8b: non-JSON stdout must raise a descriptive RuntimeError carrying a truncated
    excerpt, never a raw json.JSONDecodeError."""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProc(stdout=b"not json at all")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    model = ChatClaudeCLI(model="haiku")
    with pytest.raises(RuntimeError) as exc_info:
        await model.ainvoke("hi")

    assert not isinstance(exc_info.value, json.JSONDecodeError)
    assert "non-JSON" in str(exc_info.value)
    assert "not json at all" in str(exc_info.value)


async def test_t6b_undecodable_stdout_bytes_do_not_raise_a_unicode_error(monkeypatch) -> None:
    """M8b: undecodable bytes must decode with errors='replace' rather than raising
    UnicodeDecodeError, and still surface as the same descriptive RuntimeError."""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProc(stdout=b"\xff\xfe not valid utf-8 {")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    model = ChatClaudeCLI(model="haiku")
    with pytest.raises(RuntimeError):
        await model.ainvoke("hi")


def test_t4_stateless_argv_never_resumes() -> None:
    """T4: the wrapper never passes --resume/--continue. Unit-level, no
    network — inspects the argv _build_argv() constructs."""
    model = ChatClaudeCLI(model="opus")
    argv = model._build_argv("some prompt")

    assert "--resume" not in argv
    assert "--continue" not in argv
    assert argv[0] == model.claude_path
    assert argv[1:5] == ["-p", "some prompt", "--output-format", "json"]
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "opus"
