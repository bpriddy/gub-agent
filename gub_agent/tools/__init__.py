"""
GUB agent tool functions.

All tools are plain Python functions — ADK infers their descriptions from
docstrings and their parameter schemas from type hints.

Import pattern in agent.py:
    from .tools import ALL_TOOLS
    agent = Agent(..., tools=ALL_TOOLS)
"""

from .accounts import get_account_overview, get_campaign, list_accounts
from .discovery import find, get_idea, get_piece, list_ideas
from .org_query import org_query
from .staff import find_staff_for_resourcing, get_staff_profile, search_staff

ALL_TOOLS = [
    # Discovery — resolve a named thing to a typed id when you don't know what it is
    find,
    # Structured query primitive — preferred for filter/sort/count/aggregate
    org_query,
    # Staff & resourcing
    find_staff_for_resourcing,
    get_staff_profile,
    search_staff,
    # Accounts & campaigns (detail tools)
    list_accounts,
    get_account_overview,
    get_campaign,
    # Pieces & ideas (detail tools)
    get_piece,
    list_ideas,
    get_idea,
]

__all__ = [
    "ALL_TOOLS",
    "find",
    "org_query",
    "find_staff_for_resourcing",
    "get_staff_profile",
    "search_staff",
    "list_accounts",
    "get_account_overview",
    "get_campaign",
    "get_piece",
    "list_ideas",
    "get_idea",
]
