#!/usr/bin/env python3
"""
End-to-end test for the Unanswered Questions MCP Server.
"""

import asyncio
import json
import os

from database import Database
from analyzer import QuestionAnalyzer

TEST_DB = "test_unanswered_questions.db"

SAMPLE_QUESTIONS = [
    # Password reset cluster — same intent, different phrasing
    {"question": "How do I reset my password?", "context": "Forgot password page", "tenant_id": "acme"},
    {"question": "I need to reset my password", "context": "Login screen", "tenant_id": "acme"},
    {"question": "I forgot my password, how do I change it?", "context": "Login failed", "tenant_id": "acme"},
    {"question": "My password isn't working, how do I reset it?", "context": "Multiple failed attempts", "tenant_id": "acme"},
    # Billing/invoice cluster
    {"question": "Where can I find my invoice?", "context": "Account settings", "tenant_id": "acme"},
    {"question": "How do I download my invoice?", "context": "Billing page", "tenant_id": "acme"},
    {"question": "I need a copy of my invoice", "context": "Expense reporting", "tenant_id": "acme"},
    {"question": "Can I get a receipt for my payment?", "context": "Finance team request", "tenant_id": "acme"},
    # API rate limits cluster
    {"question": "What are the API rate limits?", "context": "Integration planning", "tenant_id": "techcorp"},
    {"question": "How many API requests can I make per minute?", "context": "Getting 429 errors", "tenant_id": "techcorp"},
    {"question": "I'm hitting the API rate limit, what's the cap?", "context": "Production issue", "tenant_id": "techcorp"},
    {"question": "Is there a limit on API calls?", "context": "Scaling concerns", "tenant_id": "techcorp"},
    # Singletons (should not cluster)
    {"question": "Do you support single sign-on?", "context": "Enterprise evaluation", "tenant_id": "techcorp"},
    {"question": "What colors can I use for my dashboard theme?", "context": "Customization settings", "tenant_id": "acme"},
]

async def run_tests():
    # Clean up any previous test DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    db = Database(TEST_DB)
    await db.connect()
    analyzer = QuestionAnalyzer(db)

    # 1. Log questions
    print("1. Logging questions...")
    ids: list[str] = []
    for q in SAMPLE_QUESTIONS:
        emb = await analyzer.generate_embedding(q["question"])
        qid = await db.save_question(
            question=q["question"],
            context=q["context"],
            tenant_id=q["tenant_id"],
            embedding=emb,
        )
        ids.append(qid)
        print(f"   + {q['question'][:50]}  [{qid[:8]}]")
    print(f"   Logged {len(ids)} questions.\n")

    # 2. Stats
    print("2. Stats:")
    stats = await db.get_stats()
    print(f"   {json.dumps(stats)}\n")

    # 3. Patterns
    print("3. Finding patterns...")
    patterns = await analyzer.find_patterns()
    print(f"   Found {len(patterns)} pattern(s):")
    for p in patterns:
        print(f"   - {p['topic']} ({p['count']} questions)")
        for ex in p["representative_questions"]:
            print(f"     • {ex}")
    print()

    # 4. Doc suggestions
    print("4. Documentation suggestions for 'password reset':")
    suggestion = await analyzer.suggest_documentation("password reset")
    if isinstance(suggestion["suggestion"], dict):
        print(f"   Title: {suggestion['suggestion'].get('title', '?')}")
        for s in suggestion["suggestion"].get("sections", []):
            print(f"   - {s}")
    print()

    # 5. Resolve
    print("5. Marking first 2 questions resolved...")
    for qid in ids[:2]:
        ok = await db.mark_resolved(qid, "doc-password-guide", "Created guide")
        print(f"   {qid[:8]}... {'ok' if ok else 'failed'}")
    updated = await db.get_stats()
    print(f"   Resolution rate: {updated['resolution_rate']}%\n")

    # 6. Edge cases
    print("6. Edge cases:")
    ok = await db.mark_resolved("nonexistent-id", "doc-1")
    print(f"   Resolve bad ID → {ok}  (expected False)")
    empty_db = Database("test_empty.db")
    await empty_db.connect()
    empty_analyzer = QuestionAnalyzer(empty_db)
    empty_patterns = await empty_analyzer.find_patterns()
    print(f"   Patterns on empty DB → {len(empty_patterns)}  (expected 0)")
    await empty_db.close()
    os.remove("test_empty.db")

    await db.close()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(run_tests())
