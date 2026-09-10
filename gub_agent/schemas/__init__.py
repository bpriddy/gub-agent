"""
schemas/ — typed wire contracts the pipeline enforces in code.

`answer.py` holds AnswerPayload, the answer contract (blend 03): the shape the
formatter agent must produce and the bot renders. Contracts live here — not in
`agents/` — because the bot and the eval tooling mirror them field for field.
"""

from .answer import (
    AnswerPayload,
    BulletBlock,
    Candidate,
    Fact,
    TableBlock,
    TextBlock,
)

__all__ = [
    "AnswerPayload",
    "BulletBlock",
    "Candidate",
    "Fact",
    "TableBlock",
    "TextBlock",
]
