import os
import json
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = clickhouse_connect.get_client(
            host=os.environ["CLICKHOUSE_HOST"],
            port=int(os.environ.get("CLICKHOUSE_PORT", 8443)),
            username=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ["CLICKHOUSE_PASSWORD"],
            database=os.environ.get("CLICKHOUSE_DB", "default"),
            secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
        )
    return _client


def create_tables():
    client = get_client()

    client.command("""
        CREATE TABLE IF NOT EXISTS ny_tenant_laws (
            id          UUID DEFAULT generateUUIDv4(),
            category    String,
            topic       String,
            statute     String,
            summary     String,
            full_text   String DEFAULT '',
            source_url  String,
            state       String DEFAULT 'NY',
            city        String DEFAULT '',
            inserted_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (category, topic, statute)
    """)

    client.command("""
        CREATE TABLE IF NOT EXISTS lease_sessions (
            session_id  String,
            lease_text  String,
            filename    String DEFAULT '',
            uploaded_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (session_id, uploaded_at)
    """)

    client.command("""
        CREATE TABLE IF NOT EXISTS redline_reports (
            report_id   UUID DEFAULT generateUUIDv4(),
            session_id  String,
            preferences String,
            findings    String,
            created_at  DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (session_id, created_at)
    """)


def upsert_laws(laws: list[dict]):
    if not laws:
        return
    client = get_client()
    rows = [
        [
            law.get("category", ""),
            law.get("topic", ""),
            law.get("statute", ""),
            law.get("summary", ""),
            law.get("full_text", ""),
            law.get("source_url", ""),
            law.get("state", "NY"),
            law.get("city", ""),
        ]
        for law in laws
    ]
    client.insert(
        "ny_tenant_laws",
        rows,
        column_names=["category", "topic", "statute", "summary", "full_text", "source_url", "state", "city"],
    )


def get_law_count() -> int:
    client = get_client()
    result = client.query("SELECT count() FROM ny_tenant_laws")
    return result.first_row[0]


def search_laws(keywords: list[str], limit: int = 5) -> list[dict]:
    if not keywords:
        return []
    client = get_client()
    conditions = " OR ".join(
        f"topic ILIKE '%{kw}%' OR category ILIKE '%{kw}%' OR summary ILIKE '%{kw}%'"
        for kw in keywords[:5]
    )
    result = client.query(
        f"SELECT category, topic, statute, summary, source_url FROM ny_tenant_laws WHERE {conditions} LIMIT {limit}"
    )
    return [
        {
            "category": row[0],
            "topic": row[1],
            "statute": row[2],
            "summary": row[3],
            "source_url": row[4],
        }
        for row in result.result_rows
    ]


def get_top_laws(limit: int = 20) -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT category, topic, statute, summary, source_url FROM ny_tenant_laws LIMIT {limit}"
    )
    return [
        {
            "category": row[0],
            "topic": row[1],
            "statute": row[2],
            "summary": row[3],
            "source_url": row[4],
        }
        for row in result.result_rows
    ]


def store_lease(session_id: str, lease_text: str, filename: str = "") -> None:
    client = get_client()
    client.insert(
        "lease_sessions",
        [[session_id, lease_text, filename]],
        column_names=["session_id", "lease_text", "filename"],
    )


def get_lease(session_id: str) -> str | None:
    client = get_client()
    result = client.query(
        "SELECT lease_text FROM lease_sessions WHERE session_id = {sid:String} ORDER BY uploaded_at DESC LIMIT 1",
        parameters={"sid": session_id},
    )
    if result.result_rows:
        return result.result_rows[0][0]
    return None


def store_report(session_id: str, preferences: dict, findings: list) -> None:
    client = get_client()
    client.insert(
        "redline_reports",
        [[session_id, json.dumps(preferences), json.dumps(findings)]],
        column_names=["session_id", "preferences", "findings"],
    )


def seed_from_file(path: str) -> int:
    with open(path) as f:
        laws = json.load(f)
    upsert_laws(laws)
    return len(laws)
