"""
tenant.py — which branded surface a turn arrived through, read from session state.

One Agent Engine serves more than one Chat bot. The bots are separate Cloud Run
services with separate Chat apps, separate databases and separate user-visible
identities, but they share this engine — so the engine has to be told which
surface a turn came through, on the turn itself. It cannot be deployment
config: a prompt variant or an env var is per-ENGINE, and both bots point at the
same one.

The bot puts the label in session state at `create_session`:

    create_session(state={"gub_jwt": ..., "tenant": "<label>"})

and this module does two things with it, both cheap and both per-call:

1. **Labels the logs.** ReasoningEngine log lines carry no bot identity, so from
   the first tenant turn every existing query — format-gate rates, the
   `dispatcher: intent` denominator, router paths — silently mixes two bots'
   traffic. `label_of` gives every such line a `tenant=` field, defaulting to
   `anomaly` so the original bot's lines gain a field rather than change
   meaning.

2. **Scopes the prompt.** `tenant_instruction` appends a block naming the
   account this surface is branded for. That is a SOFT boundary and is meant to
   be: nothing here restricts what the agent can retrieve, and the tools answer
   with whatever the signed-in user's own grants allow. It is written down as an
   accepted risk rather than implied — see the epic (gub-gchat-bot#83, #90).
   Filter-level scoping was investigated and does not exist: the Workspace
   connector data stores publish no schema, and a Vertex AI Search filter binds
   to a schema key property.

Invariants, both pinned by tests:

- With no `tenant` key in state, EVERY function here is a no-op: the prompt is
  the base provider's text, byte for byte, and the log label is `anomaly`. The
  engine runs this code on every single prod turn.
- A malformed label is a HARD error, never a silent fall back to unscoped. A bot
  that believes it is scoped and is not is the failure worth being loud about,
  and the label is already validated at the bot's own boot, so reaching here
  malformed means something is genuinely wrong.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext

STATE_KEY = "tenant"

#: The label for traffic that carries no tenant — i.e. the original bot.
DEFAULT_LABEL = "anomaly"

#: Same shape the bot validates at boot (`src/config.ts`): a lowercase slug. It
#: reaches a prompt and a log field, so it stays boring on purpose.
_LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def _state_of(ctx: Any) -> Any:
    """Session state from any context flavour: `ReadonlyContext.state`
    (instruction providers), `CallbackContext.state` (model callbacks), or
    `ctx.session.state` (InvocationContext, and the fakes in tests)."""
    state = getattr(ctx, "state", None)
    if state is not None:
        return state
    session = getattr(ctx, "session", None)
    return getattr(session, "state", None)


def read_tenant(state: Any) -> str | None:
    """The tenant label in `state`, or None when the turn carries none.

    Raises ValueError on a present-but-malformed label — see the module
    docstring on why this is loud.
    """
    if state is None:
        return None
    try:
        raw = state.get(STATE_KEY)
    except AttributeError:
        return None
    if raw is None:
        return None
    if not isinstance(raw, str) or not _LABEL.match(raw):
        raise ValueError(
            f"state[{STATE_KEY!r}] must be a lowercase slug matching "
            f"{_LABEL.pattern!r}; got {raw!r}"
        )
    return raw


def label_of(ctx_or_state: Any) -> str:
    """The tenant label for a LOG line: the turn's tenant, or `anomaly`.

    Never raises — a log call is not the place to discover a bad label, and the
    paths that matter (the instruction provider) already did.
    """
    try:
        return read_tenant(_state_of(ctx_or_state) or ctx_or_state) or DEFAULT_LABEL
    except (ValueError, AttributeError):
        return DEFAULT_LABEL


def tenant_note(tenant: str) -> str:
    """The '## Tenant surface' block appended to the instruction for a tenant turn.

    Deliberately not a refusal. The ruling (gub-gchat-bot#83, D4) is that this
    surface is *branded* for an account, not *restricted* to it, so the model
    treats the account as the default subject and SAYS SO when a question is
    plainly about something else — the "answered with a caveat" behaviour. A
    hard refusal here would read as a security boundary that does not exist:
    nothing stops the same user asking the same question through the other bot,
    or through the backend directly.

    The label is interpolated, never hardcoded — this engine is multi-company
    and a client name in a prompt is a client name in every tenant's prompt.
    """
    return (
        "## Tenant surface\n"
        f"This turn arrived through the **{tenant}** surface: a chat app branded "
        f"for the {tenant} account.\n"
        f"- Treat {tenant} as the default subject. When a question does not name "
        f"an account, it is about {tenant}.\n"
        f"- You are not restricted to {tenant}. If the question is plainly about "
        "something else, answer it normally — then add one short line noting that "
        f"this surface is the {tenant} one.\n"
        "- Never say or imply that other accounts are hidden, blocked or "
        "unavailable to you. They are not."
    )


def tenant_instruction(
    base_provider: Callable[[ReadonlyContext], str],
) -> Callable[[ReadonlyContext], str]:
    """Wrap an InstructionProvider so a tenant turn carries the scope block.

    Wraps OUTSIDE the sandbox wrapper: a sandbox run swaps which prompt is used,
    and the surface a turn arrived through is true of that run either way. With
    no tenant in state the base provider's text is returned unchanged — same
    object, no concatenation.
    """

    def provider(ctx: ReadonlyContext) -> str:
        tenant = read_tenant(_state_of(ctx))
        base = base_provider(ctx)
        if tenant is None:
            return base
        return f"{base}\n\n{tenant_note(tenant)}"

    return provider
