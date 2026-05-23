import uuid
import llm
import storage
from pydantic import BaseModel


class Finding(BaseModel):
    id: str
    type: str          # law_violation | preference_conflict | missing_protection | risky_clause
    severity: str      # RED | YELLOW | INFO
    title: str
    clause_text: str
    clause_location: str
    char_start: int
    char_end: int
    issue: str
    law_reference: str | None = None
    suggested_rewrite: str | None = None


def _build_prompt(lease_text: str, preferences: dict, laws: list[dict]) -> str:
    pref_labels = {
        "pets": "Pets allowed in unit",
        "parking": "Parking included",
        "early_termination": "Early termination clause (30-day buyout)",
        "guests": "Guests allowed longer than 7 days",
        "subletting": "Subletting allowed",
        "landlord_notice": "Landlord must give 24h notice before entry",
        "rent_increase_notice": "Rent increase notice required",
        "security_deposit": "Security deposit capped at 1 month",
        "late_fee": "Late fee capped at 5% / 5-day grace period",
        "repairs_timeline": "Explicit repair response timeline in lease",
    }
    prefs_lines = [
        f"- {label}: TENANT WANTS THIS"
        for key, label in pref_labels.items()
        if preferences.get(key) not in (None, False)
    ]
    laws_lines = [f"- {law['statute']}: {law['summary']}" for law in laws]

    return f"""You are a New York tenant rights attorney reviewing a residential lease.

TENANT PREFERENCES (flag any clause that conflicts):
{chr(10).join(prefs_lines) if prefs_lines else '- No specific preferences provided'}

NEW YORK LAW REQUIREMENTS (flag violations and missing protections):
{chr(10).join(laws_lines)}

LEASE TEXT:
{lease_text}

INSTRUCTIONS:
- Return a Finding for EVERY problematic clause.
- clause_text must be the EXACT text from the lease.
- char_start / char_end are character offsets of clause_text within the lease text above. Use -1 for missing_protection findings.
- type: law_violation | preference_conflict | missing_protection | risky_clause
- severity: RED (violates law or immediately harmful) | YELLOW (conflicts with preferences or risky) | INFO (worth noting)
- suggested_rewrite: tenant-friendly version (1-2 sentences).
- law_reference: specific statute (e.g. "NY RPL § 227-e") for law_violation findings.
- id: short unique string per finding.
"""


async def analyze(lease_text: str, preferences: dict, session_id: str):
    """Async generator — yields Finding objects one by one."""
    laws = storage.get_top_laws(limit=20)
    # Groq free tier caps at ~12k TPM; keep lease under ~28k chars (~7k tokens) to leave room for prompt overhead
    prompt = _build_prompt(lease_text[:28000], preferences, laws)

    findings: list[Finding] = await llm.generate_list(prompt, Finding, temperature=0.1)

    for f in findings:
        if not f.id:
            f.id = str(uuid.uuid4())

    storage.store_report(session_id, preferences, [f.model_dump() for f in findings])

    for f in findings:
        yield f
