"""OpenAI setup and structured model calls; no query or database logic."""
import asyncio
import json
import os
import re
from datetime import datetime
from time import perf_counter

from pydantic import BaseModel

from cynapse.settings import LLM_TIMEOUT_SECONDS, SG
from cynapse.data_models import Extraction, QueryPlan
from cynapse.prompt import PROMPT


class ExtractionError(ValueError):
    """Safe operational failure, distinct from user ambiguity."""

    def __init__(self, message: str, category="provider_error"):
        """Create an extraction failure with a safe message and machine-readable type.

        Args:
            message: User-facing text that must not contain provider secrets.
            category: Failure label, such as timeout, malformed_output or provider_error.

        Returns:
            None. Stores category and initializes the ValueError message."""
        super().__init__(message)
        self.category = category


class LLMExtractor:
    """Wrap a structured language model behind a synchronous extraction callable.

    One asyncio runner is reused across turns. last_call holds the latest status,
    latency, rather than raw model content.
    Call close() when finished to release the HTTP client and event loop."""
    def __init__(self, model, timeout: float = LLM_TIMEOUT_SECONDS, structured_options: dict | None = None):
        """Configure structured Extraction output without running inference.

        Args:
            model: LangChain model supporting with_structured_output().
            timeout: Maximum seconds per invocation; greater than zero and at most 120.
            structured_options: Optional keyword arguments for structured output setup.

        Returns:
            None. Creates a reusable runner and structured model wrapper.

        Raises:
            ValueError: Timeout is outside the allowed range.
            Exception: The supplied model cannot configure structured output."""
        if not 0 < timeout <= 120:
            raise ValueError("Timeout must be between 0 and 120 seconds")
        self.timeout = timeout
        self.model = model
        self.runner = asyncio.Runner()
        self.structured = model.with_structured_output(Extraction, include_raw=True, **(structured_options or {}))
        self.last_call = None

    def close(self):
        # Keep one loop across turns: SDK HTTP clients are bound to that loop.
        """Release resources owned by this extractor.

        Args:
            None.

        Returns:
            None. Closes the model's asynchronous HTTP client when present, then the
            reusable asyncio runner. Call after the final extraction."""
        client = getattr(self.model, "root_async_client", None)
        if client is not None:
            self.runner.run(client.close())
        self.runner.close()

    def __call__(self, text: str, previous: QueryPlan | None, now: datetime, pending: dict | None = None):
        """Extract validated intent and filter changes from one conversation turn.

        Args:
            text: User's request, kept separate from the system prompt.
            previous: Last accepted QueryPlan, or None for a new conversation.
            now: Timezone-aware clock, sent to the model in Singapore time.
            pending: Optional unresolved query/clarification context.

        Returns:
            A validated Extraction. Also replaces last_call with status and elapsed seconds.

        Raises:
            ExtractionError: Timeout, malformed output, authentication/access failure
                or another provider error; messages avoid raw provider details.

        Makes a model call (potentially paid). Does not generate or execute SQL."""
        started = perf_counter()
        self.last_call = {"status": "started"}
        explicit_reset = re.match(r"\s*(?:start over|reset|new search|clear context)\b", text, re.IGNORECASE) is not None
        payload = {
            "request": text,
            "now": now.astimezone(SG).isoformat(),
            "accepted_filters": previous.model_dump(mode="json") if previous and not explicit_reset else None,
            "pending_clarification": pending if not explicit_reset else None,
        }
        messages = [("system", PROMPT), ("human", json.dumps(payload))]

        async def invoke():
            """Run the prepared structured model call under its timeout.

            Args:
                None; uses messages and timeout from the enclosing call.

            Returns:
                The model's structured/raw output wrapper.

            Raises:
                TimeoutError: The configured deadline expires.
                Exception: Model/provider errors, handled by the enclosing call."""
            return await asyncio.wait_for(self.structured.ainvoke(messages), timeout=self.timeout)

        try:
            result = self.runner.run(invoke())
            if isinstance(result, dict) and "raw" in result and "parsed" in result:
                refusal = (getattr(result["raw"], "additional_kwargs", None) or {}).get("refusal")
                if refusal:
                    result = {
                        "intent": "reject", "reset": False, "continuation": False,
                        **{field: {"action": "keep", "value": None}
                           for field in ("camera", "date", "time", "recurrence")},
                        "clarification_question": None,
                    }
                elif result.get("parsing_error") is not None or result["parsed"] is None:
                    raise ExtractionError("The language model returned malformed filters. Please retry.", "malformed_output")
                else:
                    result = result["parsed"]
            data = result.model_dump() if isinstance(result, BaseModel) else result
            try:
                extraction = Extraction.model_validate(data)
            except ValueError as exc:
                raise ExtractionError("The language model returned malformed filters. Please retry.", "malformed_output") from exc
            complete_request = re.match(r"\s*(?:show|get|retrieve)\b", text, re.IGNORECASE)
            if (complete_request and extraction.camera.action == "set" and extraction.date.action == "set"
                    and not re.search(r"\b(?:instead|correction)\b", text, re.IGNORECASE)):
                extraction = extraction.model_copy(update={"continuation": False})
            self.last_call["status"] = "success"
            return extraction
        except ExtractionError as exc:
            self.last_call["status"] = exc.category
            raise
        except TimeoutError as exc:
            self.last_call["status"] = "timeout"
            raise ExtractionError("The language model timed out. Please retry.", "timeout") from exc
        except Exception as exc:
            # Provider errors may include credentials or raw content. Never echo them.
            if getattr(exc, "status_code", None) in (401, 403):
                self.last_call["status"] = "authentication_error"
                raise ExtractionError("Model authentication/access failed. Check your API key and model permissions.", "authentication_error") from exc
            self.last_call["status"] = "provider_error"
            raise ExtractionError("The language model could not produce a valid response. Please retry.", "provider_error") from exc
        finally:
            self.last_call["latency_seconds"] = round(perf_counter() - started, 6)


def configured_extractor(model_name: str | None = None):
    """Construct the OpenAI extractor selected by explicit name or environment.

    Args:
        model_name: Optional openai:model identifier. Falls back to CYNAPSE_MODEL.

    Returns:
        LLMExtractor using strict JSON-schema output and no automatic retries.
        The integration reads OPENAI_API_KEY; this function does not run inference.

    Raises:
        ValueError: Configuration is missing, the provider is unsupported, the
            OpenAI extra is unavailable, or model initialization fails."""
    name = model_name or os.getenv("CYNAPSE_MODEL")
    if not name:
        raise ValueError("Set CYNAPSE_MODEL to openai:model for AI chat. Use cynapse chat --nonai for the nonai version.")
    if not name.startswith("openai:") or not name.removeprefix("openai:").strip():
        raise ValueError("Only OpenAI models are supported. Use openai:model.")
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise ValueError("Install the openai extra first: pip install -e '.[openai]'.") from exc
    # No hidden retries or silent fallback to the nonai version.
    try:
        model = init_chat_model(name, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
        return LLMExtractor(model, structured_options={"method": "json_schema", "strict": True})
    except Exception as exc:
        raise ValueError("Could not initialize OpenAI. Check the openai extra, model name and API key.") from exc
