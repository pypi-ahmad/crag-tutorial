"""Agnes Chat Completions with bounded retries and secret-safe errors."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import random
import time
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from src.config import BASE_URL, MODEL_NAME

_counter = ContextVar("agnes_requests", default=None)

class ModelRequestError(RuntimeError):
    """Provider failure without response bodies or credentials."""

@contextmanager
def count_requests():
    parent = _counter.get()
    count = {"attempts": 0}
    token = _counter.set(count)
    try:
        yield count
    finally:
        _counter.reset(token)
        if parent is not None:
            parent["attempts"] += count["attempts"]

def get_openai_client():
    key = os.environ.get("AGNESAI_API_KEY")
    if not key:
        raise ModelRequestError("AGNESAI_API_KEY is missing from the process environment.")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=60, max_retries=0)

def chat(messages, model=MODEL_NAME, temperature=0, max_tokens=512, max_retries=5, backoff_base=2, client=None):
    if model != MODEL_NAME:
        raise ValueError("Only agnes-3.0-flash is supported.")
    if max_retries < 1:
        raise ValueError("max_retries must allow at least one attempt.")
    owned = client is None
    client = client or get_openai_client()
    try:
        for attempt in range(max_retries):
            count = _counter.get()
            if count is not None:
                count["attempts"] += 1
            try:
                response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                choice = response.choices[0]
                content = choice.message.content
                if not content or choice.finish_reason == "length":
                    raise ModelRequestError("Model returned empty or truncated content; no result was cached.")
                return content.strip()
            except APIError as error:
                status = getattr(error, "status_code", None)
                retryable = isinstance(error, (APIConnectionError, APITimeoutError)) or status in (408, 409, 429) or (status is not None and status >= 500)
                if not retryable or attempt + 1 == max_retries:
                    raise ModelRequestError(f"Agnes request failed: {type(error).__name__}; HTTP {status or 'unavailable'}.") from None
                delay = min(30, backoff_base * 2 ** attempt + random.random())
                response = getattr(error, "response", None)
                if response is not None:
                    try:
                        delay = max(delay, min(60, float(response.headers.get("retry-after", "0"))))
                    except ValueError:
                        pass
                time.sleep(delay)
    finally:
        if owned:
            client.close()
