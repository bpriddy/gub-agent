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
   account this surface is branded for, and tells the model what to say when a
   tool result reports that records were withheld.

   This block is NOT the boundary. Since UBScope (gub #119) the scope is a
   presentation guardrail owned by the app: the bot declares it per tenant
   (`{"account": [{"id", "name"}, ...]}`, a name beside every id) and puts it in
   session state as `app_scope`, a STRING, every turn. The agent forwards it
   untouched as the `X-App-Scope` header on every GUB call (`tools/_client.py`)
   and the backend interprets it and narrows each read; it is intersected with
   the user's grants and never widens them. The agent only READS it, here, for
   the labels: the persona names the accounts the surface covers from the
   scope's own names instead of from the tenant slug. The block stays because
   it is still the honest description of the surface, and because a filtered
   read has to be described rather than reported as an absence.

   The 📊 half is scoped; the ✉️ Workspace half is NOT, and cannot be with the
   machinery that exists: the connector data stores publish no schema, and a
   Vertex AI Search filter binds to a schema key property. Staff, offices and
   teams are unscoped by ruling — no staff→account link exists to scope them
   by. See the epic (gub-gchat-bot#83, #90).

Invariants, both pinned by tests:

- With no `tenant` key in state, EVERY function here is a no-op: the prompt is
  the base provider's text, byte for byte, and the log label is `anomaly`. The
  engine runs this code on every single prod turn.
- A tenant turn with no app scope is a HARD error too (`require_app_scope`): a
  bot that believes it is scoped and sends nothing must not run unscoped.
- A malformed label is a HARD error, never a silent fall back to unscoped. A bot
  that believes it is scoped and is not is the failure worth being loud about,
  and the label is already validated at the bot's own boot, so reaching here
  malformed means something is genuinely wrong.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext

logger = logging.getLogger(__name__)

STATE_KEY = "tenant"

#: Session-state key holding the app scope: the bot's serialized JSON, as a string.
SCOPE_STATE_KEY = "app_scope"

#: Bounds on what is read back out of the scope for the prompt (the backend
#: caps entries and label length at the same figures).
_MAX_LABELS = 50
_MAX_LABEL_CHARS = 128

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


def require_app_scope(state: Any) -> None:
    """Fail hard when a tenant turn carries no app scope.

    A tenant bot never runs unscoped: with `state["tenant"]` set, `app_scope`
    must be a non-empty string (the bot's serialized scope). Anything else —
    absent, `None`, an object, an empty string — raises ValueError, the same
    way a malformed tenant label does, and writes one `tenant_scope: missing`
    line. Only the type of the offending value is logged, never the value: it
    can hold the scope's labels.

    No tenant, no check: with no `tenant` in state this is a no-op, so every
    non-tenant turn is byte for byte what it was.

    Two call sites, because there are two doors: the tenant instruction
    provider (the deep path) and `tools/_client.py` before any request (the
    fast path never runs an instruction provider).
    """
    tenant = read_tenant(state)
    if tenant is None:
        return
    raw = state.get(SCOPE_STATE_KEY)
    if isinstance(raw, str) and raw:
        return
    logger.error(
        "tenant_scope: missing state_key=%s value_type=%s tenant=%s",
        SCOPE_STATE_KEY,
        type(raw).__name__,
        tenant,
    )
    raise ValueError(
        f"state[{STATE_KEY!r}] is set but state[{SCOPE_STATE_KEY!r}] is "
        f"{'empty' if raw == '' else 'missing' if raw is None else 'not a string'}: "
        "a tenant turn must carry its app scope"
    )


def read_scope_labels(state: Any) -> list[str]:
    """The human labels of the app scope in `state`, in declaration order.

    Read-only: parses a COPY of the string, returns names only (never ids), and
    never writes anything back. Total — a missing, non-string or unparseable
    scope yields `[]`, because the backend is what validates a scope and this is
    only the persona asking what to call the surface; callers fall back to the
    tenant slug.
    """
    if state is None:
        return []
    try:
        raw = state.get(SCOPE_STATE_KEY)
    except AttributeError:
        return []
    if not isinstance(raw, str):
        return []
    try:
        scope = json.loads(raw)
    except (ValueError, RecursionError):
        return []
    if not isinstance(scope, dict):
        return []
    labels: list[str] = []
    for entries in scope.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str):
                continue
            # One line, bounded: the label lands in a prompt.
            name = " ".join(name.split())[:_MAX_LABEL_CHARS]
            if name and name not in labels:
                labels.append(name)
            if len(labels) >= _MAX_LABELS:
                return labels
    return labels


def _names(tenant: str, labels: list[str] | None) -> str:
    """What to call the covered subject in prose: the scope's own labels
    ("A", "A and B", "A, B and C"), or the tenant slug when no scope is present."""
    if not labels:
        return tenant
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def _account_phrase(tenant: str, labels: list[str] | None) -> str:
    """'the <names> account(s)'. With no scope this is exactly the phrase the
    prompt has always carried ('the <slug> account')."""
    plural = "s" if labels and len(labels) > 1 else ""
    return f"the {_names(tenant, labels)} account{plural}"


def label_of(ctx_or_state: Any) -> str:
    """The tenant label for a LOG line: the turn's tenant, or `anomaly`.

    Never raises — a log call is not the place to discover a bad label, and the
    paths that matter (the instruction provider) already did.
    """
    try:
        return read_tenant(_state_of(ctx_or_state) or ctx_or_state) or DEFAULT_LABEL
    except (ValueError, AttributeError):
        return DEFAULT_LABEL


def tenant_note(tenant: str, labels: list[str] | None = None) -> str:
    """The '## Tenant surface' block appended to the instruction for a tenant turn.

    Still not a refusal rule, and still not the boundary — that now lives in
    the backend (tenant-10). What changed is that the old text was written for
    a surface which was *branded* and not *restricted*, and said so:
    "you are not restricted", "never say or imply that other accounts are
    unavailable". Both are now false for the 📊 half, and a model told them
    will argue with its own tool results.

    So the rule is evidence-bound rather than absolute: describe records as
    withheld exactly when a tool result says they were, and never otherwise.
    That keeps it correct in BOTH enforcement modes — in 'log' mode nothing is
    withheld, no notice appears, and this block changes nothing.

    Two failure modes it steers away from, deliberately:
      - "there are no campaigns for that account" — a claim about the world
        made from a fact about permissions, and the reason this exists;
      - "you don't have access to that" — the user's own grants are not what
        filtered it, and telling them otherwise sends them to ask for access
        they already have.

    The subject is interpolated, never hardcoded — this engine is multi-company
    and a client name in a prompt is a client name in every tenant's prompt. It
    is the scope's own labels when the turn carries a scope (`labels`, from
    `read_scope_labels`), and the tenant slug until it does.
    """
    subject = _names(tenant, labels)
    return (
        "## Tenant surface\n"
        f"This turn arrived through the **{tenant}** surface: a chat app branded "
        f"for {_account_phrase(tenant, labels)}, and scoped to it in our database.\n"
        f"- Treat {subject} as the default subject. When a question does not name "
        f"an account, it is about {subject}.\n"
        "- A tool result may carry `account_scope_notice`. When it does, some "
        "records were withheld because they are outside this surface's scope. "
        "Answer with what you DID get, and leave the withheld part alone.\n"
        "- Do not write a sentence announcing the scope. The surface says that "
        "itself, in one line above your answer; saying it again reads as a "
        "stutter, and it was rendered three times on one turn before this "
        "rule existed.\n"
        "- When records are withheld, never say the thing does not exist, and "
        "never say the user lacks access — neither is true.\n"
        "- If EVERYTHING you would have answered from was withheld, abstain. "
        "A placeholder answer that only restates the limit is worse than no "
        "section at all, because the surface has already stated it.\n"
        "- Without that notice, nothing was withheld: answer normally and do "
        "not speculate about what might have been filtered.\n"
        "- Your own mail, chat and files searches are NOT scoped. Do not "
        "describe those results as limited to one account."
    )


def tenant_note_formatter(tenant: str, labels: list[str] | None = None) -> str:
    """The formatter's version of the block.

    The formatter needs its own because it is the ONLY model on some turns:
    the fast path runs no executor at all, and smalltalk, abstain and clarify
    turns never reach one either. A rule that lives only in the executor
    instruction is therefore absent from exactly the short turns where a
    one-line notice is the whole answer.

    It is deliberately narrower than the executor's. The formatter sees no
    tool results — its input is the executor's text plus the evidence index —
    so its job is to not LOSE a notice that is already there, and to not
    invent one that is not.

    The last bullet is not decoration. `format_gate` rejects a capitalised run
    that appears in no tool result as an ungrounded entity, and a tenant label
    written into the answer is exactly such a word: it burns all three
    formatter attempts and drops the turn into the template fallback.
    """
    return (
        "## Tenant surface\n"
        f"This turn arrived through the **{tenant}** surface, which is scoped to "
        f"{_account_phrase(tenant, labels)} in our database.\n"
        "- Do not add a sentence saying records were withheld, and do not keep "
        "one if the text you were given has it: the surface states that itself, "
        "once, above your answer. Two statements of it read as a stutter.\n"
        "- Never restate it as the records not existing, and never as the user "
        "lacking access.\n"
        "- Do not add the surface's name to the answer when it does not appear "
        "in the evidence you were given."
    )


def tenant_instruction(
    base_provider: Callable[[ReadonlyContext], str],
    note: Callable[..., str] = tenant_note,
    tenant_base: Callable[[ReadonlyContext], str] | None = None,
) -> Callable[[ReadonlyContext], str]:
    """Wrap an InstructionProvider so a tenant turn carries the scope block.

    Wraps OUTSIDE the sandbox wrapper: a sandbox run swaps which prompt is used,
    and the surface a turn arrived through is true of that run either way. With
    no tenant in state the base provider's text is returned unchanged — same
    object, no concatenation.

    `tenant_base`, when given, REPLACES the base provider on a tenant turn (the
    executor's tenant persona); it never runs without a tenant.
    """

    def provider(ctx: ReadonlyContext) -> str:
        tenant = read_tenant(_state_of(ctx))
        if tenant is None:
            return base_provider(ctx)
        require_app_scope(_state_of(ctx))
        base = (tenant_base or base_provider)(ctx)
        labels = read_scope_labels(_state_of(ctx))
        return f"{base}\n\n{note(tenant, labels)}"

    return provider
