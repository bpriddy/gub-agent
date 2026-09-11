"""
schemas/ — typed wire contracts the pipeline enforces in code.

`answer.py` holds AnswerPayload, the answer contract (blend 03): the shape the
formatter agent must produce and the bot renders. Contracts live here — not in
`agents/` — because the bot and the eval tooling mirror them field for field.

`router.py` holds RouterDecision, the routing contract (blend 04). It sits
here for the same reason of shape-over-wiring, with one difference worth
knowing: nothing outside the engine reads it, so renaming a field there is not
a cross-repo wire change.
"""

from .answer import (
    AnswerPayload,
    BulletBlock,
    Candidate,
    Fact,
    TableBlock,
    TextBlock,
)
from .router import FAST_INTENTS, Intent, RouterDecision

__all__ = [
    "FAST_INTENTS",
    "AnswerPayload",
    "BulletBlock",
    "Candidate",
    "Fact",
    "Intent",
    "RouterDecision",
    "TableBlock",
    "TextBlock",
]
