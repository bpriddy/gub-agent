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


def list_accounts(
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
    return gub_get("/org/accounts", tool_context, q=query, limit=limit)


def get_account_overview(
    account_id: str,
    tool_context: Any = None,
) -> dict:
    """
    Get a full overview of a client account: details plus all campaigns.

    Returns the account record together with every campaign associated with it,
    including campaign name, status, dates, and the staff who created each one.

    Use this when asked about a specific client, their campaign history,
    or what work the agency has done for them. Requires the account UUID —
    use list_accounts first if you only have a name.

    Args:
        account_id: The UUID of the account

    Returns:
        dict with account details and a nested 'campaigns' list.
        Each campaign includes id, name, status, createdAt, and createdByStaff.
    """
    account = gub_get(f"/org/accounts/{account_id}", tool_context)
    if account.get("error"):
        return account

    campaigns_resp = gub_get(f"/org/accounts/{account_id}/campaigns", tool_context)
    campaigns = (
        campaigns_resp.get("campaigns", campaigns_resp)
        if isinstance(campaigns_resp, dict) and not campaigns_resp.get("error")
        else []
    )

    return {**account, "campaigns": campaigns}


def get_campaign(
    campaign_id: str,
    tool_context: Any = None,
) -> dict:
    """
    Get details of a single campaign.

    Returns campaign name, status, dates, the parent account, and the staff
    member who created it.

    Use this when you have a specific campaign UUID and need its full details.
    Use get_account_overview to browse all campaigns for an account.

    Args:
        campaign_id: The UUID of the campaign

    Returns:
        dict with campaign details including account and createdByStaff.
    """
    return gub_get(f"/org/campaigns/{campaign_id}", tool_context)
