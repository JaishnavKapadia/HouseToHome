import llm
import storage
from pydantic import BaseModel


class LawReference(BaseModel):
    statute: str
    summary: str
    source_url: str


class TenantAnswer(BaseModel):
    lease_excerpt: str | None
    lease_clause_number: str | None
    law: LawReference
    steps: list[str]
    severity: str   # "HIGH" | "MEDIUM" | "LOW"
    emergency: bool


class ClauseResult(BaseModel):
    clause_text: str | None
    clause_location: str | None


async def _extract_keywords(question: str) -> list[str]:
    return await llm.generate_string_list(
        f"Extract 3-5 legal topic keywords from this tenant housing question. "
        f"Focus on housing law topics like: heat, repairs, deposit, eviction, pets, subletting, entry, fees, rent.\n\n"
        f"Question: {question}",
        temperature=0.0,
    )


async def _find_relevant_clause(question: str, lease_text: str) -> tuple[str, str] | None:
    prompt = (
        f"Find the single most relevant clause from this lease for the tenant's question.\n\n"
        f"Question: {question}\n\n"
        f"Lease:\n{lease_text[:30000]}\n\n"
        f"Return clause_text (exact lease text, max 300 chars) and clause_location "
        f"(e.g. 'Section 12, Paragraph 2'). If no relevant clause exists, return null for both."
    )
    result = await llm.generate_obj(prompt, ClauseResult, temperature=0.1)
    if result and result.clause_text:
        return (result.clause_text, result.clause_location or "")
    return None


async def _compose_answer(
    question: str,
    laws: list[dict],
    clause: tuple[str, str] | None,
) -> TenantAnswer | None:
    laws_block = "\n".join(
        f"- {law['statute']}: {law['summary']}" for law in laws
    ) or "Use general NY tenant law knowledge."

    clause_block = ""
    if clause:
        clause_block = f"\nLEASE CLAUSE:\n  Location: {clause[1]}\n  Text: {clause[0]}\n"

    prompt = (
        "You are a New York tenant rights attorney. A tenant has a problem.\n"
        "Use the NY laws and lease clause below to give a precise, actionable response.\n\n"
        f"TENANT QUESTION: {question}\n\n"
        f"RELEVANT NY LAWS:\n{laws_block}\n"
        f"{clause_block}\n"
        "Provide:\n"
        "- lease_excerpt: exact relevant lease text (or null)\n"
        "- lease_clause_number: section reference (or null)\n"
        "- law: the single most applicable statute with statute code, plain-English summary, and source URL\n"
        "- steps: 4-7 numbered action steps ordered by urgency\n"
        "- severity: HIGH (immediate health/safety), MEDIUM (significant violation), or LOW (minor)\n"
        "- emergency: true only if action needed within 24-48 hours\n"
    )
    return await llm.generate_obj(prompt, TenantAnswer, temperature=0.3)


async def answer(question: str, session_id: str | None = None) -> dict:
    keywords = await _extract_keywords(question)
    laws = storage.search_laws(keywords, limit=5)
    if not laws:
        laws = storage.get_top_laws(limit=5)

    lease_text = None
    if session_id:
        lease_text = storage.get_lease(session_id)

    clause = None
    if lease_text:
        clause = await _find_relevant_clause(question, lease_text)

    result = await _compose_answer(question, laws, clause)
    return result.model_dump() if result else {"error": "Could not generate answer"}
