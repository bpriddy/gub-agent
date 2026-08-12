"""
accounts.py — Client account and campaign tools for the GUB agent.

Tools:
  list_accounts        — discover accounts the user can access
  get_account_overview — account details + full campaign history
  get_campaign         — single campaign deep-dive
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get


async def list_accounts(
    query: str | None = None,
    limit: int = 20,
    tool_context: Any = None,
) -> dict:
    """
    List client accounts the authenticated user has access to.

    Returns accounts that the user's GUB access grants allow them to see.
    Useful for discovering which clients the agency works with, or for
    finding an account UUID when you only know its name.

    Examples:
    - "what accounts do I have access to?" → no args
    - "find the Nike account" → query="Nike"

    Args:
        query: Filter by account name (contains match, case-insensitive)
        limit: Maximum results (default 20)

    Returns:
        dict with 'accounts' list. Each entry has id, name, and parent info
        for sub-accounts.
    """
    return await gub_get("/org/accounts", tool_context, q=query, limit=limit)


async def get_account_overview(
    account_id: str,
    tool_context: Any = None,
) -> dict:
    """
    Get a full overview of a client account: details plus all campaigns.

    Returns the account record with a nested `campaigns` list. Each campaign
    carries name, status, budget, dates, and a truncated `statusSummary` — the
    account detail embeds these in ONE fetch (backend `GET /org/accounts/:id`),
    so there is no separate per-campaign call.

    Use this when asked about a specific client, their campaign history, or what
    work the agency has done for them. Requires the account UUID — use
    list_accounts first if you only have a name.

    The nested campaigns ALREADY carry `status` and a `statusSummary` for every
    campaign. For an ASSESSMENT question ("how is this account doing", "what's
    the status", a health/overview read), that is SUFFICIENT — synthesize your
    answer from this response alone and do NOT call `get_campaign` per campaign.
    Call `get_campaign` ONLY when the user asks about ONE specific campaign in
    depth (its full `statusMarkdown` writeup or pieces) — never as a blanket
    follow-up across every campaign in the list.

    Args:
        account_id: The UUID of the account

    Returns:
        dict with account details and a nested 'campaigns' list. Each campaign
        includes id, name, status, budget, dates, and statusSummary.
    """
    # The account detail already embeds campaigns[] (with statusSummary) — one
    # fetch, no separate campaigns-by-account call. Requires the backend that
    # embeds the stubs to be deployed (companion backend PR).
    account = await gub_get(f"/org/accounts/{account_id}", tool_context)
    if account.get("error"):
        return account
    account.setdefault("campaigns", [])
    return account


async def get_campaign(
    campaign_id: str,
    tool_context: Any = None,
) -> dict:
    """
    Get details of a single campaign.

    Returns campaign name, status, dates, the parent account, the staff
    member who created it, and a pre-written Markdown status summary
    (`statusMarkdown`) when one has been synthesized by the Drive
    review-approval flow.

    Use this when you have a specific campaign UUID and need its full
    details. Use get_account_overview to browse all campaigns for an
    account.

    Args:
        campaign_id: The UUID of the campaign

    Returns:
        dict with campaign details. When the `statusMarkdown` key is
        present and non-empty, render it VERBATIM in your response —
        it's hand-shaped status prose meant for direct display, not a
        summary input. `pieces` lists the campaign's execution pieces as
        STUBS ordered newest-first (name, jobNumber, dates — no status);
        fetch full detail for the relevant ones via get_piece, in parallel.
    """
    return await gub_get(f"/org/campaigns/{campaign_id}", tool_context)
