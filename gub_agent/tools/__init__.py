"""
GUB agent tool functions.

All tools are plain Python functions — ADK infers their descriptions from
docstrings and their parameter schemas from type hints.

Import pattern in agent.py:
    from .tools import ALL_TOOLS
    agent = Agent(..., tools=ALL_TOOLS)
"""

from .accounts import get_account_overview, get_campaign, list_accounts
from .staff import find_staff_for_resourcing, get_staff_profile, search_staff

ALL_TOOLS = [
    # Staff & resourcing
    find_staff_for_resourcing,
    get_staff_profile,
    search_staff,
    # Accounts & campaigns
    list_accounts,
    get_account_overview,
    get_campaign,
]

__all__ = [
    "ALL_TOOLS",
    "find_staff_for_resourcing",
    "get_staff_profile",
    "search_staff",
    "list_accounts",
    "get_account_overview",
    "get_campaign",
]
