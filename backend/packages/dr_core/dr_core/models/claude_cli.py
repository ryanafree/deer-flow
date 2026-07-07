"""ChatClaudeCLI — tool-free ``claude -p`` synthesis/arbitration model plane.

Per DECISIONS.md D1 (Fable consult 01): this wrapper drives ONLY synthesis/
arbitration nodes in the DeerFlow fork, as a text-in/text-out, stateless
``BaseChatModel``. It never binds tools (``bind_tools`` raises
``NotImplementedError``) and never passes ``--resume``/``--continue`` — every
call is a fresh, one-shot ``claude -p`` subprocess.

Empirical ``claude -p`` JSON contract (verified against Claude Code 2.1.199):
``claude -p '<prompt>' --output-format json --model <haiku|sonnet|opus>``
prints ONE JSON object to stdout with (at least) ``.result`` (str),
``.usage.{input_tokens,output_tokens,cache_creation_input_tokens,
cache_read_input_tokens}`` (ints), ``.total_cost_usd`` (float), ``.is_error``
(bool), ``.subtype`` (str), ``.session_id`` (str). Cold latency ~7s.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import threading
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Collapse a LangChain message list into a single prompt string.

    This model plane is tool-free and text-in/text-out (DECISIONS.md D1), so
    there is no structured multi-turn wire format to preserve. System and
    human content are concatenated in order; any other message types
    (AIMessage/ToolMessage history a caller chooses to pass) are rendered as
    labeled turns so prior context is still visible, but no session state is
    ever sent to the CLI itself (see ``_build_argv`` — no ``--resume``).
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, (SystemMessage, HumanMessage)):
            parts.append(str(msg.content))
        else:
            role = msg.__class__.__name__.replace("Message", "")
            parts.append(f"[{role}]: {msg.content}")
    return "\n\n".join(p for p in parts if p)


def _run_coro_sync(coro: Any) -> Any:
    """Run an async coroutine from a sync call site.

    Used only by the fallback ``_generate``; the async path (``_agenerate``)
    is primary and is what DeerFlow's graphs actually call. If no event loop
    is running, just ``asyncio.run`` it. If a loop IS already running in this
    thread (e.g. a sync tool invoked from inside an async framework), run the
    coroutine on a private event loop in a separate thread instead of trying
    to reuse the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class ChatClaudeCLI(BaseChatModel):
    """LangChain ``BaseChatModel`` backed by a headless ``claude -p`` subprocess.

    Tool-free, stateless, text-in/text-out. Async-first: ``_agenerate`` spawns
    ``claude -p ... --output-format json`` via ``asyncio.create_subprocess_exec``
    (never blocking subprocess IO — DeerFlow's Blockbuster gate trips on that).
    On wall-clock timeout or task cancellation, the whole process group is
    killed (``start_new_session=True`` + ``killpg``) so no ``claude`` process
    ever outlives the call.
    """

    model: str = "opus"
    """``claude -p --model`` value: alias (haiku|sonnet|opus) or full model id."""

    request_timeout: float = 600.0
    """Wall-clock timeout (seconds) for a single claude -p call."""

    claude_path: str = "claude"
    """Path to (or name of) the claude CLI binary."""

    extra_args: list[str] = Field(default_factory=list)
    """Additional argv appended after --model <model>. Never --resume/--continue."""

    model_config = {"arbitrary_types_allowed": True}

    _last_proc: asyncio.subprocess.Process | None = PrivateAttr(default=None)
    """Most recently spawned subprocess handle; exposed for tests/observability
    (e.g. verifying no orphan process survives a cancelled call)."""

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "request_timeout": self.request_timeout}

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Never implemented on purpose.

        ChatClaudeCLI is the tool-free synthesis/arbitration model plane
        (DECISIONS.md D1): headless Claude Code wants to execute tools
        itself, so emitting unexecuted tool_calls back to LangGraph's
        ToolNode is fragile. Route tool-bound roles through the OpenRouter
        shim instead.
        """
        raise NotImplementedError("ChatClaudeCLI is a tool-free text-in/text-out model plane (DECISIONS.md D1) and never binds tools. Use the OpenRouter ChatOpenAI shim for tool-bound roles.")

    def _build_argv(self, prompt: str) -> list[str]:
        """Build the claude -p argv. Unit-testable without spawning a process.

        Never includes --resume or --continue (D1: stateless one-shot only).
        """
        argv = [
            self.claude_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
        ]
        argv.extend(self.extra_args)
        return argv

    async def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (TimeoutError, ProcessLookupError):
            pass

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        argv = self._build_argv(prompt)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # setsid, so killpg() below can reach the whole group
            cwd=tempfile.gettempdir(),  # never the project tree (M8a): avoids reloading its CLAUDE.md/hooks per call
        )
        self._last_proc = proc

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.request_timeout)
        except TimeoutError:
            await self._kill_process_group(proc)
            raise TimeoutError(f"claude -p timed out after {self.request_timeout}s") from None
        except asyncio.CancelledError:
            # Caller (e.g. asyncio.wait_for(ainvoke(...), timeout=N)) cancelled us.
            # Kill the subprocess group before propagating so nothing is orphaned.
            await self._kill_process_group(proc)
            raise

        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exited {proc.returncode}: {stderr.decode(errors='replace')[:2000]}")

        decoded = stdout.decode(errors="replace")
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p returned non-JSON stdout: {decoded[:2000]!r}") from e
        return self._parse_result(payload)

    def _parse_result(self, payload: dict[str, Any]) -> ChatResult:
        text = payload.get("result", "")
        usage = payload.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)

        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "cache_read": cache_read,
                "cache_creation": cache_creation,
            },
        }

        response_metadata = {
            "total_cost_usd": payload.get("total_cost_usd"),
            "is_error": payload.get("is_error"),
            "subtype": payload.get("subtype"),
            "session_id": payload.get("session_id"),
            "usage": usage,
            "model": self.model,
        }

        ai_message = AIMessage(
            content=text,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Minimal sync fallback. The async path (_agenerate) is primary;
        this exists only so non-async callers still work."""
        return _run_coro_sync(self._agenerate(messages, stop=stop, run_manager=None, **kwargs))
