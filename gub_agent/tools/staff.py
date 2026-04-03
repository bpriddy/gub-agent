"""
staff.py — Staff and resourcing tools for the GUB agent.

Tools:
  find_staff_for_resourcing  — skill/interest/metadata search for staffing briefs
  get_staff_profile          — full profile for a known staff member
  search_staff               — general name/title/email search
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get


def find_staff_for_resourcing(
    query: str | None = None,
    metadata_type: str | None = None,
    metadata_label: str | None = None,
    metadata_value: str | None = None,
    office_id: str | None = None,
    status: str = "active",
    featured_only: bool = False,
    limit: int = 10,
    tool_context: Any = None,
) -> dict:
    """
    Find staff members who match a resourcing brief.

    Searches staff by skills, interests, highlights, certifications, and
    other structured metadata. Returns matching staff with their full profile
    including office and team memberships.

    Use this when asked to:
    - Find people for a project or client brief
    - Identify who has a specific skill or experience
    - Discover people in a particular office with certain expertise

    Examples:
    - "find someone with brand strategy experience" → metadata_label="Brand Strategy"
    - "who knows React in the London office?" → metadata_label="React", office_id=<london_id>
    - "find a senior designer with featured skills" → query="designer", featured_only=True

    Args:
        query: Keyword matched against staff names, titles, emails, or metadata labels/values
        metadata_type: Category to search within — e.g. "skill", "interest", "highlight", "certification"
        metadata_label: Specific metadata label — e.g. "React", "Brand Strategy" (contains match)
        metadata_value: Value to match within a metadata entry (contains match)
        office_id: Restrict to a specific office UUID (use search to find office IDs)
        status: Employment status — "active" (default), "former", or "leave"
        featured_only: Only return staff where the matched metadata entry is featured/highlighted
        limit: Maximum results (1–50, default 10)

    Returns:
        dict with 'count' and 'staff' list. Each staff entry includes profile,
        matched metadata entries, office, and team memberships.
    """
    params: dict[str, Any] = {
        "status": status,
        "limit": limit,
    }
    if query:         params["q"] = query
    if metadata_type:  params["type"] = metadata_type
    if metadata_label: params["label"] = metadata_label
    if metadata_value: params["value"] = metadata_value
    if office_id:      params["officeId"] = office_id
    if featured_only:  params["featured"] = "true"

    return gub_get("/org/resourcing", tool_context, **params)


def get_staff_profile(
    staff_id: str,
    tool_context: Any = None,
) -> dict:
    """
    Get the complete profile of a staff member including all metadata.

    Returns: personal details, role, status, start date, office, team
    memberships, and all associated metadata entries (skills, interests,
    highlights, certifications, notes).

    Use this when you need a detailed view of one specific person and
    already know their UUID. Use search_staff first if you only have a name.

    Args:
        staff_id: The UUID of the staff member

    Returns:
        dict with full staff profile and a 'metadata' list of entries.
    """
    profile = gub_get(f"/org/staff/{staff_id}", tool_context)
    if profile.get("error"):
        return profile

    metadata = gub_get(f"/org/staff/{staff_id}/metadata", tool_context)
    metadata_list = (
        metadata.get("metadata", metadata)
        if isinstance(metadata, dict) and not metadata.get("error")
        else []
    )

    return {**profile, "metadata": metadata_list}


def search_staff(
    query: str | None = None,
    status: str = "active",
    limit: int = 20,
    tool_context: Any = None,
) -> dict:
    """
    Search for staff members by name, title, email, or department.

    Returns a list of matching staff members with their core profile info
    (name, title, email, office, status). Does not include metadata entries —
    use get_staff_profile for the full picture on a specific person.

    Prefer find_staff_for_resourcing when searching by skill or capability.
    Use this for general people discovery — "who works in the Sydney office",
    "find someone called Alex", "list all strategists".

    Args:
        query: Text matched against names, titles, emails, and departments
        status: "active" (default), "former", "leave" — or omit for active only
        limit: Maximum results (default 20)

    Returns:
        dict with 'staff' list and pagination metadata.
    """
    return gub_get("/org/staff", tool_context, q=query, status=status, limit=limit)
