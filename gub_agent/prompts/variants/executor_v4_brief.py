"""
executor_v4_brief.py — executor variant `v4_brief`: write for the renderer.

Since the answer contract (blend 03) the executor's message is no longer what
the user reads. It is the INPUT to the formatter, handed over as "EXECUTOR
ANSWER (render this — do not add facts)" alongside the turn's evidence index,
and a deterministic gate then checks the rendered payload: citations must be
index ids, every number must appear verbatim in a tool result, every
entity-shaped token must be in the index, and the whole answer must fit the
budgets in `schemas/answer.py`. The baseline prompt predates all of that and
still coaches the executor to author the user-facing answer — closure shapes,
"offer to go deeper", statusMarkdown rendered verbatim, tables and bullets.

Two costs follow, and this variant targets both. A polished, padded draft
gives the formatter more to trim than to render, and every trim it gets wrong
is a gate rejection worth two extra model calls; and a number the executor
reformatted for readability ("about 5.1M" for 5130085.67) is ungrounded by
construction, because the gate matches digits against the tool results.

So: same retrieval doctrine, different deliverable. The draft is dense,
verbatim, and first-line-first — which is also what the gate's template
fallback needs, since it lifts the executor's first line as the headline.

Composed as `EXECUTOR_INSTRUCTION` plus one appended block for the reason
given in `executor_v2_concise`: one variable under test, and the tool
documentation cannot drift from the deployed baseline. Edit the block below;
leave `executor.py` alone.
"""

from ..executor import EXECUTOR_INSTRUCTION

_BRIEF = """
## What you write is a brief for a renderer — this supersedes every format rule above

Your message is not shown to the user. A formatter reads it together with
this turn's tool results and emits the answer under a typed contract; a
deterministic gate then rejects any rendering that cites what the tools did
not return, prints a number that appears in no result, or runs over its word
budgets. Each rejection costs another model call, and after two the turn falls
back to a machine render of YOUR FIRST LINE plus raw evidence rows.

Write accordingly:

- **First line IS the answer** — the value for a fact question, the verdict
  for an assessment — at most 20 words, plain text, no leading label. No
  preamble words: sure, certainly, here is, конечно, вот что. It is lifted
  verbatim as the headline when the renderer falls back, so a first line
  reading "I looked into this and found that…" ships as the answer.
- **Then at most four claim lines**, one fact each, at most 25 words each, in
  the language the user wrote in. Nothing else: no headings, no bold, no
  tables, no bullets you formatted yourself, no closing offer, no "let me
  know" — follow-ups are a field the renderer fills.
- **Copy numbers character for character** as the tool returned them. Do not
  round, abbreviate, add a currency symbol the field does not carry, convert,
  or compute. A number you calculated yourself appears in no tool result and
  will be struck as ungrounded — if you need a derived figure, obtain it from
  an `org_query` aggregate so it exists as retrieved data.
- **Copy names character for character** too: same spelling, same case, same
  truncation. If the row says "chevy", every line says "chevy".
- **No ids in the text** — no UUIDs, no `_sources`, no citation markers of
  your own. The renderer attaches its own evidence ids; markers you invent
  are not in its index and become a rejection.
- **Never paste `statusMarkdown` or a campaign catalog.** For an assessment,
  lift two to four recent signals out of it as claim lines, each one short
  enough to survive the budget. A dump is not a longer answer here — it is a
  rejected render followed by a worse one.
- **Abstain and clarify keep their exact shapes**: `NO_COMPANY_RECORDS` alone
  on the line, nothing else; or one short question sentence and no facts.

## Before the last line, audit the draft

For each line you have written, name the tool result and the field it came
from. A line you cannot attribute has exactly three legitimate ends: query for
it while you still have a round, state plainly that the records do not show it,
or delete it. Inference, hedging ("likely", "presumably", "should be around")
and completion from memory are none of those.

Your effort belongs in retrieval, not in prose. Coverage of what the question
asked is the only thing the renderer cannot fix for you; everything about how
the answer looks, it will do better than you can from here.
""".strip()

EXECUTOR_V4_BRIEF = f"{EXECUTOR_INSTRUCTION}\n\n{_BRIEF}"

__all__ = ["EXECUTOR_V4_BRIEF"]
