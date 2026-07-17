"""
org_query.py — structured query tool for the GUB org data.

A single tool that exposes the GUB query engine (POST /org/query) with
discrete operator modules. Use this for any question that requires
filtering, sorting, counting, grouping, or aggregating across campaigns,
accounts, or staff. Prefer this over fetching lists and reasoning over
them with prose — the LLM is bad at counting and sorting; the database
is good at it.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_post


def org_query(
    entity: str,
    filter: dict | None = None,
    sort: list[dict] | None = None,
    group_by: list[str] | None = None,
    aggregate: dict | None = None,
    limit: int | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Execute a structured query against GUB org data. THE preferred tool for
    any question involving filtering, counting, sorting, ranking, or
    aggregation — the database does these reliably; do NOT do them yourself.

    ## Entities

    - "campaigns" — agency work for a client account
    - "pieces" — campaign pieces: distinct produced/producing executions within
      a campaign (a commercial, social series, tool, activation). Query these to
      RESOLVE a piece by name (`{name: {similar_to: "..."}}`) or list a campaign's
      pieces (`{campaignId: {eq: "..."}}`); the rich status is NOT here — fetch
      it with get_piece(id). Usually easier: use `find` when you don't yet know
      the thing is a piece.
    - "accounts" — clients of the agency
    - "staff" — people at the agency

    Use a separate call per entity. For multi-entity questions, chain
    queries (see "Composition" below).

    ## Filter operators

    Each filter is `{field: {op: value}}`. ONE operator per field per call.
    AND across multiple fields. For OR/range, use `in` or `between`.

      eq          {status: {eq: "active"}}              equals
      neq         {status: {neq: "lost"}}               not equals
      in          {status: {in: ["active", "won"]}}     value in list
      gt, gte     {budget: {gte: 100000}}               >, >=
      lt, lte     {budget: {lt: 50000}}                 <, <=
      between     {awardedAt: {between: ["2025-01-01","2025-12-31"]}}
      like        {name: {like: "%nike%"}}              SQL ILIKE substring
      similar_to  {name: {similar_to: "chevy"}}         pg_trgm fuzzy match —
                                                        matches Chevrolet, Chevy Trucks, etc.
                                                        USE THIS for fuzzy name lookups.
                                                        v1 limitation: must be the SOLE
                                                        filter; chain follow-ups by `id.in`.
      is_null     {endsAt: {is_null: true}}             field IS / IS NOT NULL

    ## Sort

    `[{field: "budget", direction: "desc"}, ...]`. Multi-key allowed.
    May also sort by an aggregate output name (see Aggregate).

    ## Group-by + aggregate (analytics)

    `group_by: ["status"]` groups results by that field.
    `aggregate: {<outputName>: {op: "count"|"sum"|"avg"|"min"|"max", field?: "budget"}}`

    `count` takes no field. `sum/avg/min/max` require a numeric or date field.
    Group-by without explicit aggregate implicitly counts.

    Result rows shape: `{<group_field>: value, <outputName>: number, ...}`.

    ## Limit + total

    `limit` caps the returned rows (default 50, max 100). The response includes
    `total` — the REAL DB count of matching rows independent of `limit`. USE
    `total` FOR ANY "HOW MANY" QUESTION. Never count items in the `results`
    array yourself.

    ## Composition for multi-entity questions

    org_query handles ONE entity per call. For multi-entity questions, chain
    queries using `in` as the join primitive:

      "Most expensive campaign for Chevy this year"
      1) org_query(entity="accounts", filter={name: {similar_to: "chevy"}}, limit=5)
         → pick the right accountId from the candidates
      2) org_query(
           entity="campaigns",
           filter={
             accountId: {eq: <that account's id>},
             awardedAt: {between: ["2025-01-01","2025-12-31"]}
           },
           sort=[{field: "budget", direction: "desc"}],
           limit=1
         )

      "Staff who led campaigns over $1M for auto accounts"
      1) accounts where industry=auto → ids A
      2) campaigns where accountId in A and budget > 1M → distinct createdBy ids S
      3) staff where id in S → names

    ## Worked examples

      # "How many campaigns for Chevy?"
      org_query(entity="accounts", filter={name: {similar_to: "chevy"}}, limit=5)
      # ... pick Chevrolet's id ...
      org_query(entity="campaigns",
                filter={accountId: {eq: "..."}},
                aggregate={count: {op: "count"}})
      # → results: [{count: 7}], total: 1

      # "Top 5 accounts by campaign count"
      org_query(entity="campaigns",
                group_by=["accountId"],
                aggregate={campaignCount: {op: "count"}},
                sort=[{field: "campaignCount", direction: "desc"}],
                limit=5)

      # "Total budget awarded last year"
      org_query(entity="campaigns",
                filter={awardedAt: {between: ["2025-01-01","2025-12-31"]}},
                aggregate={total: {op: "sum", field: "budget"}})

      # "Active campaigns sorted by budget"
      org_query(entity="campaigns",
                filter={status: {eq: "active"}},
                sort=[{field: "budget", direction: "desc"}],
                limit=20)

    ## What this tool deliberately does NOT do

    - No native joins. Use chained queries with `in` as shown above.
    - No text search over status markdown / notes / metadata. Those are
      conversational context — fetch the specific entity by id with the
      existing detail tools (get_campaign / get_account_overview) and read.
    - No staff metadata (skills, certifications). Use
      `find_staff_for_resourcing` for resourcing questions.

    Args:
        entity: One of "campaigns", "accounts", "staff".
        filter: Map of field -> {operator: value}. Optional.
        sort: List of {field, direction: "asc"|"desc"}. Optional.
        group_by: List of fields to group rows by. Optional.
        aggregate: Map of output-name -> {op, field?}. Optional.
        limit: Max rows to return (default 50, max 100). Optional.

    Returns:
        {results: [...], total: int, truncated: bool}
        - results: the row list (entity rows or aggregate/group rows).
        - total: the REAL count of matching rows in the DB, NOT len(results).
        - truncated: True if limit cut off some rows.
    """
    body: dict = {"entity": entity}
    if filter is not None:
        body["filter"] = filter
    if sort is not None:
        body["sort"] = sort
    if group_by is not None:
        body["group_by"] = group_by
    if aggregate is not None:
        body["aggregate"] = aggregate
    if limit is not None:
        body["limit"] = limit
    return gub_post("/org/query", body, tool_context)
