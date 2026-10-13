"""
Reader-facing copy of the deterministic lookup-failure answers.

The no-access and not-found payloads are shown to the reader as they are, so
they use the reader's vocabulary ("company records"), never a system name or
the permission model behind it.
"""

from __future__ import annotations

import re

import pytest

from gub_agent.agents.answers import no_access_payload, not_found_payload

_NO_ACCESS_BODY = {
    "en": "You don't have access to this record yet — ask its owner to share it.",
    "ru": "У вас пока нет доступа к этой записи — попросите её владельца открыть вам доступ.",
}


def _texts(payload) -> list[str]:
    return [payload.headline, *[b.text for b in payload.blocks]]


@pytest.mark.parametrize("language", ["en", "ru"])
def test_no_access_body_is_the_neutral_copy(language):
    payload = no_access_payload(language)

    assert [b.text for b in payload.blocks] == [_NO_ACCESS_BODY[language]]


def test_no_access_unknown_language_falls_back_to_english():
    assert no_access_payload("de").blocks[0].text == _NO_ACCESS_BODY["en"]  # type: ignore[arg-type]


def test_no_access_russian_body_is_all_russian():
    # A Latin letter in the Russian copy is a leaked system name or English.
    assert re.search(r"[A-Za-z]", no_access_payload("ru").blocks[0].text) is None


@pytest.mark.parametrize("language", ["en", "ru"])
@pytest.mark.parametrize("make", [no_access_payload, not_found_payload])
def test_lookup_failure_copy_names_no_system_or_grants(make, language):
    for text in _texts(make(language)):
        lowered = text.lower()
        assert "gub" not in lowered
        assert "grant" not in lowered
        assert "грант" not in lowered
