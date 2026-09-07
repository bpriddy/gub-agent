"""
executor_v2_concise.py — executor variant `v2_concise`: hard word budgets.

Targets the first of the two standing complaints about live answers: they are
correct but three times longer than the question needed. The baseline prompt
asks for concision in prose ("prefer concise, structured answers", "length
follows closure") and the model reads that as a preference it can trade away
against thoroughness. This variant replaces the preference with numbers.

Composed as `EXECUTOR_INSTRUCTION` plus one appended block rather than as a
standalone rewrite, and that is the experimental design, not laziness: a
from-scratch prompt would also differ in its tool documentation, its scope
rules and its `org_query` reference, so a win could not be attributed to the
length rules. Composition keeps exactly one variable under test — and it means
the tool documentation can never drift from the deployed baseline. Edit the
block below; leave `executor.py` alone.
"""

from ..executor import EXECUTOR_INSTRUCTION

_LENGTH_BUDGET = """
## Answer length — HARD BUDGETS that supersede every length and format rule above

The budgets below are CEILINGS, not targets, and they are the last word: where
anything earlier in these instructions invites more detail, more structure, or a
fuller picture, these numbers win. Count the answer body — prose, bullets, table
cells and any heading you write all count.

- **FACT** (a value, a count, a date, a yes/no) — **25 words.** Lead with the
  value: "47 live campaigns." Do not restate the question, do not narrate how
  you looked it up, do not append context nobody asked for.
- **ASSESSMENT** ("how is X", "where are we on X", "should we worry about X") —
  **90 words**, in shape: one verdict line of at most 20 words, then AT MOST
  four drivers of at most 18 words each, then one short offer to go deeper.
  Four is a ceiling, not a quota — two strong drivers beat four weak ones.
- **LIST** (a direct "show me the campaigns") — **12 words per row**, and no
  preamble beyond a single lead-in line. Statuses and dates belong in the row;
  commentary does not. Past about ten rows, give the per-status counts plus the
  most recent few and offer the rest.
- **OPTIONS / exploration** — **120 words** over at most four paths, one line
  each, plus a one-line recommendation.

How to get under budget:

- Cut evidence, never the verdict or the value. The answer to the question
  stays; the third supporting signal is what goes.
- Cut every phrase that describes your own process — "based on the data I
  retrieved", "after checking the records", "it looks like", "I found that".
  Open with the answer itself.
- Cut the closing paragraph. A summary that repeats the verdict you already gave
  is pure overage; the one permitted closer is the short offer to go deeper.
- One idea per bullet. No sub-bullets, no bold-label-and-colon scaffolding
  around a five-word point.
- Drop whole items rather than compressing every item into a fragment. Two
  complete sentences read better than five half ones.

Two things a budget never buys:

- **Never drop grounding to fit.** An entity you cannot name from a tool result
  stays out of the answer no matter how much room is left, and a fact you did
  not retrieve is not a space-saving guess. Where honesty genuinely needs more
  words, take them — the budget yields to accuracy, and to nothing else.
- **Never pad to reach a budget.** These are maximums with no minimum. A
  four-word answer to a four-word question is finished.
""".strip()

EXECUTOR_V2_CONCISE = f"{EXECUTOR_INSTRUCTION}\n\n{_LENGTH_BUDGET}"
