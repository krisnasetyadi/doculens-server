"""
The "feature guide" — the only knowledge the conversation mode is allowed
to answer from (see intent.py / processor.generate_conversation_answer).

User-facing features only: what a user can see and do in DocuLens. It must
never describe how DocuLens is built (frameworks, model/provider, database,
hosting, prompts) — guard.py refuses those questions, and keeping them out
of this text means the LLM has nothing to leak even if the guard misses one.

Plan details mirror chat-ui's lib/pricing-plans.ts (prices/limits shown on
the pricing page) — update both together. Token quotas mirror
router/payment.py PLAN_QUOTAS, but are deliberately left out: users see
token usage through /usage and the live snapshot passed per request, not
through hardcoded numbers that can drift.
"""

from typing import Any, Dict, List, Optional

DOCULENS_FEATURE_GUIDE = """\
ABOUT DOCULENS
DocuLens is a workspace where users ask questions about their OWN data and get answers grounded in it, with source citations (file name and page).

SOURCES (Sources panel in the sidebar)
- PDF / documents: upload files (or type /upload). Files are grouped into collections and can be organized in nested folders.
- Web / Drive links: add a public link (e.g. a shared Google Drive folder or web page) as a live source.
- Database: connect an external database; its tables become a searchable source. Admin only.
- Chat logs: upload WhatsApp/chat exports, or connect Telegram and sync chosen chats. Admin only.
- Before asking about data, at least one source must be switched on in the chat composer.

ASKING QUESTIONS
- Ask in plain language, e.g. "summarize my document", "what does clause 5 say?", "who handles invoices?". Answers cite the sources used.
- Follow-up questions work: the chat remembers the last few messages of the session.
- Chat sessions are saved and can be reopened from the sidebar.
- Efficient Mode (experimental) compresses context to use fewer tokens; /efficiency shows the comparison.

SLASH COMMANDS (type "/" in the chat box to see them)
- /gap-check — Compliance Gap Check (Individual and Team plans only — NOT on Free): compare company documents against any standard/framework document you uploaded (e.g. ISO 27001, ISO 9001). Result: status per item (met / partial / not met / unknown) with evidence and recommendations, downloadable as a PDF/markdown report. On the Free plan it's not available — suggest upgrading to Individual or Team; don't suggest /gap-check to users whose current plan is Free.
- /collections — list your document collections.
- /history — previous gap-analysis runs.
- /upload — upload a new document.
- /usage — your token usage summary.
- /efficiency — Efficient Mode token comparison.
- /help — list all commands.
- Skills: users can upload their own Skills (custom instructions); each one gets its own slash command and shapes the next answer.

TEAMS
- On the Team plan an admin invites members, assigns which sources each member can access, and allocates each member's share of the workspace token quota. Members can request more tokens from their admin.

PLANS (prices per month)
- Free — Rp 0: 1 user, PDF source only, 20 queries per month, 200 MB storage, personal use.
- Individual — Rp 65.000: 1 user, all source types (PDF, Database, Chat, Web Link), 5 GB storage, Compliance Gap Check (/gap-check), personal search history.
- Team — Rp 500.000: 5 members + 1 admin seat, admin controls source access per member, one shared workspace, 30 GB shared storage, Compliance Gap Check (/gap-check), centralized billing, priority support.
- Enterprise — custom pricing: unlimited seats, custom storage, SSO & advanced access control, custom integrations (SharePoint, Google Drive, MongoDB), SLA & dedicated support, on-premise / private cloud deployment. Contact sales.
- Upgrade from the pricing page. Usage resets every 30-day subscription period.
- There is also a short-term safety rate limit; if it's reached, the chat box is paused until the shown reset time.
"""


def build_user_state_block(
    is_admin: bool,
    collection_titles: List[str],
    collection_count: int,
    usage: Optional[Dict[str, Any]] = None,
    has_source: bool = True,
) -> str:
    """Live facts about THIS user, appended after the static guide so the
    model can answer "how many tokens do I have left?" or suggest a next
    step using their real collections — never invented numbers."""
    lines = [
        f"- Role: {'admin' if is_admin else 'member'}",
        f"- Sources switched on for this message: {'yes' if has_source else 'none'}",
    ]
    if collection_count:
        names = ", ".join(collection_titles[:5])
        more = ", …" if collection_count > 5 else ""
        lines.append(f"- Document collections: {collection_count} ({names}{more})")
    else:
        lines.append("- Document collections: none yet (they should upload via the Sources panel or /upload)")

    if usage:
        if usage.get("plan_name"):
            lines.append(f"- Current plan: {usage['plan_name']}")
        if usage.get("my_token_limit") is not None:
            lines.append(
                f"- Your tokens this period: {usage.get('my_token_used', 0):,} used of "
                f"{usage['my_token_limit']:,}"
            )
        elif usage.get("my_token_used") is not None:
            lines.append(f"- Your tokens used this period: {usage['my_token_used']:,}")
        if usage.get("workspace_token_limit") is not None:
            lines.append(
                f"- Workspace tokens this period: {usage.get('workspace_token_used', 0):,} used of "
                f"{usage['workspace_token_limit']:,}"
            )
        if usage.get("period_end"):
            lines.append(f"- Current period ends: {usage['period_end']}")
    else:
        lines.append("- Usage numbers: not available right now (point them to /usage)")

    return "USER STATE\n" + "\n".join(lines)
