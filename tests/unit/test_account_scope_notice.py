"""
test_account_scope_notice.py — the out-of-scope signal, agent side (tenant-10).

A scoped GUB session filters another account's records out of a read. Without
a signal on the response, that filtered read is indistinguishable from an
empty one, and a model reporting honestly on what it received says "Budweiser
has no campaigns" — a claim about the world assembled from a fact about
permissions. Worse than a refusal, and the reason this plumbing exists.

Two things are pinned here that are easy to get subtly wrong:

  * the notice key must NOT start with "_" — underscore-prefixed keys are
    stripped from the model's view by context pruning and skipped by the
    evidence index, so the notice would be attached and never seen;
  * a bare JSON array cannot carry a notice at all (the evidence index drops
    non-dict payloads), so those responses are wrapped — under the key that
    endpoint's existing readers already use, not a generic one that would
    break them.
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.tools._client import (
    ACCOUNT_SCOPE_NOTICE,
    _array_key,
    _with_scope_notice,
)


def _resp(filtered: bool) -> SimpleNamespace:
    headers = {"x-account-scope-filtered": "1"} if filtered else {}
    return SimpleNamespace(headers=headers)


def test_unfiltered_response_is_untouched() -> None:
    payload = {"results": [1, 2]}
    assert _with_scope_notice(payload, _resp(False), "/org/query") is payload


def test_dict_payload_carries_the_notice() -> None:
    out = _with_scope_notice({"results": []}, _resp(True), "/org/query")
    assert out["account_scope_notice"] == ACCOUNT_SCOPE_NOTICE


def test_the_notice_key_is_visible_to_the_model() -> None:
    out = _with_scope_notice({"results": []}, _resp(True), "/org/query")
    keys = [k for k in out if "scope" in k]
    assert keys == ["account_scope_notice"]
    assert not any(k.startswith("_") for k in keys)


def test_bare_array_is_wrapped_so_it_can_carry_one() -> None:
    hits = [{"id": "1", "type": "account"}]
    out = _with_scope_notice(hits, _resp(True), "/org/search")
    assert out["hits"] == hits
    assert out["account_scope_notice"] == ACCOUNT_SCOPE_NOTICE


def test_the_wrapper_key_is_the_one_that_endpoint_s_readers_use() -> None:
    # fast_path reads response["hits"]; a generic "results" would make a
    # filtered search look like no match at all, which is the failure this
    # whole file is about, one layer down.
    assert _array_key("/org/search") == "hits"
    assert _array_key("/org/accounts") == "accounts"
    assert _array_key("/org/campaigns/abc") == "results"


def test_the_notice_forbids_both_wrong_answers() -> None:
    lowered = ACCOUNT_SCOPE_NOTICE.lower()
    assert "non-existent" in lowered  # not "there is no such account"
    assert "lack access" in lowered  # not "you don't have permission"
    assert "not available from this surface" in lowered
