import os
import re
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

import llm
import storage


LAW_SOURCES = [
    "https://www.nyc.gov/site/hpd/renters/tenants-rights.page",
    "https://ag.ny.gov/tenants-rights-guide",
    "https://hcr.ny.gov/tenant-rights",
    "https://www.nyc.gov/site/hpd/services-and-information/heat-hot-water.page",
    "https://www.lawhelp.org/ny/resource/new-york-tenants-rights-guide",
]

NIMBLE_API_KEY = os.environ.get("NIMBLE_API_KEY", "")


class LawEntry(BaseModel):
    category: str   # repairs, security_deposit, eviction, habitability, fees, entry, pets, etc.
    topic: str      # snake_case identifier
    statute: str    # e.g. "NY RPL § 235-b"
    summary: str    # plain-English 1-3 sentence summary
    source_url: str


async def _nimble_fetch(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.webit.live/api/v1/realtime/web",
                headers={
                    "Authorization": f"Bearer {NIMBLE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"url": url, "render": True, "country": "US", "locale": "en"},
            )
            resp.raise_for_status()
            return resp.json().get("html_content") or ""
    except Exception as e:
        print(f"Nimble fetch failed for {url}: {e}")
        return None


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


async def _parse_laws_from_html(html: str, source_url: str) -> list[LawEntry]:
    page_text = _html_to_text(html)[:40000]

    prompt = (
        "You are a New York tenant rights legal researcher. "
        "Extract every distinct tenant law, right, or requirement from this page text.\n"
        "For each entry provide:\n"
        "- category: one of [habitability, repairs, security_deposit, rent, entry, eviction, fees, pets, subletting, termination, discrimination, other]\n"
        "- topic: short snake_case identifier (e.g. heat_requirement)\n"
        "- statute: the specific statute cited (e.g. 'NY RPL § 235-b'). Use 'General' if none.\n"
        "- summary: plain-English 1-3 sentence explanation\n"
        f"- source_url: '{source_url}'\n\n"
        "Return ONLY distinct, concrete legal requirements. Aim for 5-15 entries.\n\n"
        f"PAGE TEXT:\n{page_text}"
    )

    try:
        laws = await llm.generate_list(prompt, LawEntry, temperature=0.1)
        if not laws:
            print(f"  LLM returned empty list for {source_url}")
        return laws
    except Exception as e:
        print(f"  LLM error for {source_url}: {e}")
        return []


async def seed_from_nimble() -> dict:
    results = {"sources_scraped": 0, "sources_failed": 0, "laws_added": 0, "details": []}

    for url in LAW_SOURCES:
        print(f"Nimble: fetching {url}")
        html = await _nimble_fetch(url)

        if not html:
            results["sources_failed"] += 1
            results["details"].append({"url": url, "status": "fetch_failed", "count": 0})
            continue

        print(f"Nimble: parsing {len(html)} chars from {url}")
        laws = await _parse_laws_from_html(html, url)

        if laws:
            storage.upsert_laws([law.model_dump() for law in laws])
            results["laws_added"] += len(laws)
            results["sources_scraped"] += 1
            results["details"].append({"url": url, "status": "ok", "count": len(laws)})
            print(f"  -> inserted {len(laws)} laws")
        else:
            results["sources_failed"] += 1
            results["details"].append({"url": url, "status": "parse_empty", "count": 0})

    results["total_law_count"] = storage.get_law_count()
    return results
