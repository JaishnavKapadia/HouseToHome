"""
Unified LLM abstraction. Switch providers with LLM_PROVIDER env var:
  LLM_PROVIDER=groq    (default) — free, Llama 3.3 70B via Groq
  LLM_PROVIDER=claude  — Anthropic Claude (paid, use for demo)
  LLM_PROVIDER=gemini  — Google Gemini 2.5 Flash (requires billing)
"""
import json
import os
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def provider_label() -> str:
    p = provider()
    return {
        "groq":   "Groq / Llama 3.3 (free)",
        "claude": "Claude Haiku (Anthropic)",
        "gemini": "Gemini 2.5 Flash (Google)",
    }.get(p, p)


# ── Groq ─────────────────────────────────────────────────────────────────────

async def _groq_json(prompt: str, temperature: float = 0.1) -> str:
    from groq import AsyncGroq
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    resp = await client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond with valid JSON only. No markdown, no explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return resp.choices[0].message.content or "{}"


# ── Claude ───────────────────────────────────────────────────────────────────

async def _claude_json(prompt: str, temperature: float = 0.1) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=4096,
        system="You are a helpful assistant. Always respond with valid JSON only. No markdown fences, no explanation.",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.content[0].text


# ── Gemini ───────────────────────────────────────────────────────────────────

async def _gemini_list(prompt: str, item_class: Type[T], temperature: float = 0.1) -> list[T]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = await client.aio.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[item_class],
            temperature=temperature,
        ),
    )
    return resp.parsed or []


async def _gemini_obj(prompt: str, obj_class: Type[T], temperature: float = 0.3) -> T | None:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = await client.aio.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=obj_class,
            temperature=temperature,
        ),
    )
    return resp.parsed


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _extract_list(raw: str, item_class: Type[T]) -> list[T]:
    """Parse JSON text into a list of Pydantic objects robustly."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[llm] JSON parse error: {raw[:300]}")
        return []

    # If it's already a list
    if isinstance(data, list):
        return _coerce_list(data, item_class)

    # If it's an object, find the first list value
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return _coerce_list(v, item_class)

    return []


def _coerce_list(items: list, item_class: Type[T]) -> list[T]:
    result = []
    for item in items:
        try:
            result.append(item_class(**item) if isinstance(item, dict) else item_class.model_validate(item))
        except Exception as e:
            print(f"[llm] item coerce error: {e}")
    return result


def _extract_obj(raw: str, obj_class: Type[T]) -> T | None:
    try:
        data = json.loads(raw)
        return obj_class(**data)
    except Exception as e:
        print(f"[llm] obj parse error: {e}, raw: {raw[:300]}")
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def _list_prompt_suffix(item_class: Type[T]) -> str:
    schema = item_class.model_json_schema()
    return (
        f'\n\nReturn ONLY a JSON object with a single key "items" containing an array of objects. '
        f'Each object must match this schema: {json.dumps(schema, indent=None)}\n'
        f'Example: {{"items": [{{...}}, {{...}}]}}'
    )


def _obj_prompt_suffix(obj_class: Type[T]) -> str:
    schema = obj_class.model_json_schema()
    return f'\n\nReturn ONLY a valid JSON object matching this schema: {json.dumps(schema, indent=None)}'


async def generate_list(prompt: str, item_class: Type[T], temperature: float = 0.1) -> list[T]:
    """Generate a list of Pydantic objects from the active provider."""
    p = provider()

    if p == "gemini":
        return await _gemini_list(prompt, item_class, temperature)

    augmented = prompt + _list_prompt_suffix(item_class)

    if p == "claude":
        raw = await _claude_json(augmented, temperature)
    else:
        raw = await _groq_json(augmented, temperature)

    return _extract_list(raw, item_class)


async def generate_obj(prompt: str, obj_class: Type[T], temperature: float = 0.3) -> T | None:
    """Generate a single Pydantic object from the active provider."""
    p = provider()

    if p == "gemini":
        return await _gemini_obj(prompt, obj_class, temperature)

    augmented = prompt + _obj_prompt_suffix(obj_class)

    if p == "claude":
        raw = await _claude_json(augmented, temperature)
    else:
        raw = await _groq_json(augmented, temperature)

    return _extract_obj(raw, obj_class)


async def generate_string_list(prompt: str, temperature: float = 0.0) -> list[str]:
    """Generate a plain list of strings."""
    p = provider()

    if p == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = await client.aio.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt + '\n\nReturn ONLY a JSON array of strings, e.g. ["heat", "repairs"]',
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[str],
                temperature=temperature,
            ),
        )
        return resp.parsed or []

    augmented = prompt + '\n\nReturn ONLY a JSON object: {"items": ["string1", "string2", ...]}'

    if p == "claude":
        raw = await _claude_json(augmented, temperature)
    else:
        raw = await _groq_json(augmented, temperature)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
    except Exception:
        pass
    return []
