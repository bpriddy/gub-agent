"""
files.py — Drive file-name search for the GUB agent.

Tools:
  find_files  — find a Drive file by describing its NAME

Until search-01 there was exactly ONE way the bot could find a file: the
Workspace spoke, which asks Gemini Enterprise and ranks inside a black box we
cannot see into or tune. Meanwhile GUB has held a Drive index all along
(`drive_file_snapshots`, ~26k files, every row carrying its account) that
nothing was reading. This is the second, LOCAL path — trigram over the names
GUB already has, with a score we can print and a corpus we can measure offline.

It ADDS a route, it does not replace one. The index holds names, not document
text, so a question about what is INSIDE a file is unanswerable here by
construction, and the Workspace half stays the only thing that can answer it.
That asymmetry is why an empty result from this tool means "not found by name"
and never "no such file" — the docstring below is where that instruction
reaches the model.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get


# The docstring below deliberately does NOT tell the model to SAY "we could not
# find it by name". That sentence was in all three prompts (here, the executor
# and the critic) and it is undeliverable: an empty result leaves the turn with
# no citable evidence, and an evidence-free turn is steered to `kind="abstain"`
# by `agents/format_gate.py`, which the bot renders as no company-records
# section at all — so the sentence we promised reaches nobody. Abstaining is
# the RIGHT outcome (search-01 §4/§5.2: declining and letting the Workspace
# half answer beats guessing), so the prompts describe what happens instead of
# promising prose the gate will drop. Restoring "say exactly that" would only
# re-create the mismatch; the load-bearing half is the anti-substitution rule,
# which stays.
async def find_files(
    query: str,
    account_id: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Find a Drive FILE by describing its NAME — a local search over the names of
    the Drive files GUB indexes, ranked by how much of your query the name
    actually carries.

    Reach for this instead of `find` when the question is WHICH FILE: the user
    describes a deck, a cut, an image or a document — "the BHAC 30 second
    teaser", "the Silverado HD strategic brief" — and wants the file itself.
    `find` resolves org ENTITIES (accounts, campaigns, pieces, ideas, staff) and
    returns no files at all; this returns files and no entities. It matches FILE
    NAMES ONLY: it cannot tell you what a document SAYS, what is in a deck, or
    who appears in a video. Never answer a question about a file's CONTENTS from
    a name that happens to look right.

    Examples:
    - "do we have the BHAC 30 second teaser?" → find_files(query="BHAC 30 second teaser")
    - "find the Silverado HD strategic brief" → find_files(query="Silverado HD strategic brief")
    - "any Trax assets for chevy?" → find_files(query="Trax", account_id="<chevy uuid>")

    Args:
        query: The DESCRIBING PHRASE, in the user's own words — not their whole
            sentence. Keep every word that describes the file ("BHAC 30 second
            teaser", not "BHAC teaser"): the scoring already discounts filler,
            and trimming the description is what turns an honest "not found"
            into a confident wrong file. But leave the conversation outside it.
            Measured against the live index: "can you pull up the 30 second
            teaser for BHAC Hits the Road?" finds the file, while the same
            request wrapped in "hey, I was looking for ... can you find it for
            me please?" matches NOTHING — every candidate falls under the
            similarity floor, and the search silently returns empty.
        account_id: UUID of an account to scope the search to (resolve it with
            `find` or `list_accounts`). Omit it when the question names no client.

    Returns:
        dict with a `files` list, each { fileId, name, mimeType, accountId,
        campaignId, modifiedTime, similarity, coverage }. `coverage` is the
        share of the query's distinctive words the name carries, `similarity`
        the raw trigram score; the list is already ranked and already filtered,
        so the top entries are the answer — do not re-rank it yourself.
        AN EMPTY LIST MEANS THE FILE WAS NOT FOUND BY NAME, WHICH IS NOT THE
        SAME AS THE FILE NOT EXISTING. Name no file and offer no near-match as
        a substitute. An empty list is also no evidence, so the turn has
        nothing to cite and will abstain — that is the right ending for it: we
        only index names, and a separate system searches the user's own
        Workspace by CONTENT and may well find this one.
    """
    hits = await gub_get("/org/files/search", tool_context, q=query, accountId=account_id)

    # An error keeps the shape every other tool's failures have — the model is
    # already taught to read `{"error": true, message}`, and dressing it up as
    # an empty file list would read to it as "no such file", which is the one
    # wrong answer this whole tool exists to avoid.
    if isinstance(hits, dict) and hits.get("error"):
        return hits

    # Two shapes arrive here and the model must not have to tell them apart.
    # GUB answers with a BARE ARRAY of hits; but when the tenant-10 account
    # scope filters rows out, `_client._with_scope_notice` wraps that array so
    # the notice has somewhere to live — under `_ARRAY_KEYS["/org/files/search"]`,
    # which is this same "files" key. Normalising both means the response shape
    # does not change under the reader's feet on a scoped turn.
    #
    # `files`, and `fileId` / `name` / `mimeType` on each hit, are a CROSS-REPO
    # CONTRACT: gub-gchat-bot reads them straight off this function's response,
    # keyed on the tool name `find_files`. Renaming any of them empties the
    # bot's file section with no error anywhere — rename them only in step with
    # that repo.
    if isinstance(hits, list):
        return {"files": hits}
    if isinstance(hits, dict):
        hits.setdefault("files", [])
        return hits
    return {"files": []}
