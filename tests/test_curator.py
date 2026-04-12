"""Tests for sunder.curator -- LLM-assisted curation."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from sunder.curator import (
    _format_goals,
    _is_heading,
    _split_into_sections,
    _strip_code_fences,
    apply_curation,
    curate_text,
    load_report,
    save_report,
)
from sunder.types import SunderConfig


def _mock_anthropic_response(payload: dict | str):
    """Build a fake Anthropic response whose .content[0].text is the payload."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _install_fake_anthropic(client) -> None:
    """Register a fake `anthropic` module whose Anthropic(...) returns `client`."""
    mod = ModuleType("anthropic")
    mod.Anthropic = lambda *a, **kw: client  # type: ignore[attr-defined]
    sys.modules["anthropic"] = mod


def _config_with_goals() -> SunderConfig:
    return SunderConfig(
        curate=True,
        curation_goals={"purpose": "test", "keep": ["syntax"]},
        anthropic_api_key="sk-test",
    )


def test_keep_decision_returns_original_text():
    section = "This is a syntax definition with useful content."
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "KEEP",
        "reason": "useful",
        "chunks": [section],
    })

    _install_fake_anthropic(fake_client)
    report = curate_text(section, "doc1", cfg)

    assert len(report) == 1
    assert report[0]["decision"] == "KEEP"
    assert report[0]["chunks"] == [section]


def test_split_decision_returns_multiple_chunks():
    section = "Topic A details.\n\nTopic B details."
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "SPLIT",
        "reason": "two topics",
        "chunks": ["Topic A details.", "Topic B details."],
    })

    _install_fake_anthropic(fake_client)
    report = curate_text(section, "doc1", cfg)

    assert report[0]["decision"] == "SPLIT"
    assert len(report[0]["chunks"]) == 2


def test_rewrite_decision_returns_rewritten_not_original():
    section = "orig  garbled   text"
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "REWRITE",
        "reason": "cleanup",
        "chunks": ["clean rewritten text"],
    })

    _install_fake_anthropic(fake_client)
    report = curate_text(section, "doc1", cfg)

    assert report[0]["decision"] == "REWRITE"
    assert report[0]["chunks"] == ["clean rewritten text"]
    assert report[0]["original"] == section


def test_discard_decision_returns_nothing():
    section = "irrelevant boilerplate"
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "DISCARD",
        "reason": "not relevant",
        "chunks": [],
    })

    _install_fake_anthropic(fake_client)
    report = curate_text(section, "doc1", cfg)

    assert report[0]["decision"] == "DISCARD"
    assert report[0]["chunks"] == []
    assert apply_curation(report) == []


def test_strip_code_fences_json_tag():
    wrapped = '```json\n{"decision": "KEEP", "chunks": ["x"]}\n```'
    assert _strip_code_fences(wrapped) == '{"decision": "KEEP", "chunks": ["x"]}'


def test_strip_code_fences_bare_fence():
    wrapped = '```\n{"a": 1}\n```'
    assert _strip_code_fences(wrapped) == '{"a": 1}'


def test_strip_code_fences_unfenced_passthrough():
    raw = '{"decision": "KEEP"}'
    assert _strip_code_fences(raw) == raw


def test_strip_code_fences_with_leading_whitespace():
    wrapped = '   ```json\n{"k": "v"}\n```   '
    assert _strip_code_fences(wrapped) == '{"k": "v"}'


def test_fenced_claude_response_is_parsed_not_warned():
    """A fenced response must be parsed -- not fall through to the warning."""
    section = "some body text"
    cfg = _config_with_goals()

    fenced = '```json\n{"decision": "REWRITE", "reason": "clean", "chunks": ["rewritten body"]}\n```'
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response(fenced)
    _install_fake_anthropic(fake_client)

    report = curate_text(section, "doc1", cfg)

    assert report[0]["decision"] == "REWRITE"
    assert report[0]["chunks"] == ["rewritten body"]
    assert "WARNING" not in report[0]["reason"]


def test_malformed_json_falls_back_to_keep_with_warning():
    section = "some content"
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response("not json at all {{{")

    _install_fake_anthropic(fake_client)
    report = curate_text(section, "doc1", cfg)

    assert report[0]["decision"] == "KEEP"
    assert "WARNING" in report[0]["reason"]
    assert report[0]["chunks"] == [section]


def test_apply_curation_skips_discard_records():
    report = [
        {"decision": "KEEP", "chunks": ["a"]},
        {"decision": "DISCARD", "chunks": []},
        {"decision": "SPLIT", "chunks": ["b", "c"]},
        {"decision": "REWRITE", "chunks": ["d"]},
    ]
    assert apply_curation(report) == ["a", "b", "c", "d"]


def test_save_and_load_report_round_trip(tmp_path):
    report = [
        {
            "section_id": 0,
            "doc_id": "d",
            "decision": "KEEP",
            "reason": "ok",
            "original": "hello",
            "chunks": ["hello"],
        }
    ]
    path = tmp_path / "report.json"
    save_report(report, path)
    loaded = load_report(path)
    assert loaded == report


def test_format_goals_handles_all_four_keys():
    goals = {
        "purpose": "index SQL docs",
        "keep": ["syntax", "examples"],
        "discard": ["marketing"],
        "rewrite_instructions": ["strip headers"],
    }
    out = _format_goals(goals)
    assert "Purpose: index SQL docs" in out
    assert "Keep:" in out and "- syntax" in out and "- examples" in out
    assert "Discard:" in out and "- marketing" in out
    assert "Rewrite instructions:" in out and "- strip headers" in out


def test_split_into_sections_respects_max_chars_and_paragraph_boundaries():
    p1 = "a" * 200
    p2 = "b" * 200
    p3 = "c" * 200
    text = f"{p1}\n\n{p2}\n\n{p3}"

    sections = _split_into_sections(text, max_chars=250)

    # Each paragraph is 200 chars; max 250 -> can't fit 2 together (400 > 250)
    assert len(sections) == 3
    assert sections[0] == p1
    assert sections[1] == p2
    assert sections[2] == p3


def test_split_into_sections_groups_small_paragraphs():
    p1 = "one two three"
    p2 = "four five six"
    sections = _split_into_sections(f"{p1}\n\n{p2}", max_chars=2500)
    assert len(sections) == 1
    assert "one" in sections[0] and "four" in sections[0]


def test_split_into_sections_splits_oversized_paragraph_on_single_newlines():
    # One paragraph with many lines, total > max_chars
    lines = [f"line {i} " + ("x" * 50) for i in range(40)]
    big_para = "\n".join(lines)  # single paragraph -- no double newlines inside
    assert len(big_para) > 500

    sections = _split_into_sections(big_para, max_chars=500)

    assert len(sections) > 1
    # Every line must appear somewhere, in order, without being broken up.
    joined = " ".join(sections)
    for i in range(40):
        assert f"line {i}" in joined


def test_split_into_sections_never_exceeds_twice_max_chars():
    # Mix of normal paragraphs and one oversized paragraph.
    small = "short para"
    oversized_lines = [("y" * 200) for _ in range(20)]  # 20 * 200 = 4000 chars
    oversized = "\n".join(oversized_lines)
    text = f"{small}\n\n{oversized}\n\n{small}"

    max_chars = 800
    sections = _split_into_sections(text, max_chars=max_chars)

    for s in sections:
        assert len(s) <= max_chars * 2, f"section of {len(s)} chars exceeds 2x limit"


def test_split_into_sections_handles_huge_chapter():
    # Simulate a ~100K-char chapter: many paragraphs separated by \n\n.
    paragraphs = [("word " * 80).strip() for _ in range(300)]  # ~300 * 400 = 120K
    text = "\n\n".join(paragraphs)
    assert len(text) > 100_000

    sections = _split_into_sections(text, max_chars=2500)

    assert len(sections) > 20
    for s in sections:
        assert len(s) <= 5000  # 2 * max_chars


def test_curate_text_does_not_call_apply_curation():
    """curate_text() must produce the raw report only, never apply it."""
    import sunder.curator as curator_mod

    section = "content goes here"
    cfg = _config_with_goals()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "DISCARD",
        "reason": "nope",
        "chunks": [],
    })
    _install_fake_anthropic(fake_client)

    called = {"n": 0}
    real_apply = curator_mod.apply_curation

    def spy(report):
        called["n"] += 1
        return real_apply(report)

    curator_mod.apply_curation = spy
    try:
        report = curate_text(section, "doc1", cfg)
    finally:
        curator_mod.apply_curation = real_apply

    assert called["n"] == 0
    # The raw report still contains the original, even though decision=DISCARD.
    assert report[0]["original"] == section
    assert report[0]["decision"] == "DISCARD"


# -- heading-based splitting tests -------------------------------------------


def test_is_heading_true_for_uppercase_headings():
    for h in ["CHAR", "DATE", "DEFINE RECORD", "ALTER LOG", "COLLECT"]:
        assert _is_heading(h), f"expected True for {h!r}"


def test_is_heading_false_for_mixed_case_and_titles():
    for h in ["Result", "Example", "Syntax", "mixed Case", "Some Title Here", ""]:
        assert not _is_heading(h), f"expected False for {h!r}"


def test_split_on_headings_two_sections():
    text = "CHAR\nchar content line 1\nchar content line 2\nDATE\ndate content line 1"
    sections = _split_into_sections(text, max_chars=5000)
    assert len(sections) == 2
    assert sections[0].startswith("CHAR")
    assert sections[1].startswith("DATE")


def test_split_on_headings_define_record():
    text = "DEFINE RECORD\nrecord stuff\nmore record\nDEFINE UPDATE\nupdate stuff"
    sections = _split_into_sections(text, max_chars=5000)
    assert len(sections) == 2
    assert "DEFINE RECORD" in sections[0]
    assert "DEFINE UPDATE" in sections[1]


def test_no_headings_falls_back_to_char_splitting():
    # No uppercase headings — should fall back to paragraph/char splitting
    p1 = "a" * 200
    p2 = "b" * 200
    p3 = "c" * 200
    text = f"{p1}\n\n{p2}\n\n{p3}"
    sections = _split_into_sections(text, max_chars=250)
    assert len(sections) == 3


def test_oversized_heading_section_further_split():
    # One heading section that exceeds max_chars * 2
    big_content = "\n\n".join(["para " + "x" * 300 for _ in range(10)])
    text = f"BIGHEADING\n{big_content}"
    max_chars = 500
    sections = _split_into_sections(text, max_chars=max_chars)
    assert len(sections) > 1
    for s in sections:
        assert len(s) <= max_chars * 2, f"section of {len(s)} chars exceeds 2x limit"


def test_heading_split_no_section_exceeds_twice_max():
    headings = ["ALPHA", "BRAVO", "CHARLIE"]
    parts = []
    for h in headings:
        parts.append(h)
        # Use multiple lines so oversized sections can be split on newlines
        parts.append("\n".join(["x" * 80 for _ in range(10)]))
    text = "\n".join(parts)
    max_chars = 300
    sections = _split_into_sections(text, max_chars=max_chars)
    for s in sections:
        assert len(s) <= max_chars * 2


# -- ingest() curation gate tests -------------------------------------------

from sunder.ingest import ingest


def test_ingest_raises_when_curate_true_without_report_path():
    cfg = SunderConfig(
        curate=True,
        curation_goals={"purpose": "x"},
        anthropic_api_key="sk-test",
    )
    with __import__("pytest").raises(ValueError, match="curation_report_path"):
        ingest(docs=["some raw text"], output_dir="unused", config=cfg)


def test_ingest_stops_after_saving_report_when_file_missing(tmp_path):
    cfg = SunderConfig(
        curate=True,
        curation_goals={"purpose": "test"},
        anthropic_api_key="sk-test",
    )
    report_path = tmp_path / "report.json"

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response({
        "decision": "KEEP",
        "reason": "ok",
        "chunks": ["some raw text"],
    })
    _install_fake_anthropic(fake_client)

    result = ingest(
        docs=["some raw text"],
        output_dir=str(tmp_path / "out"),
        config=cfg,
        curation_report_path=str(report_path),
    )

    assert result is None
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved[0]["decision"] == "KEEP"
    assert saved[0]["original"] == "some raw text"
    # The corpus output dir was never created -- embedding did not run.
    assert not (tmp_path / "out").exists()


def test_ingest_proceeds_when_report_exists(tmp_path, monkeypatch):
    """When curation_report_path exists, ingest loads it and proceeds past
    curation without calling Claude. We stub out downstream stages to keep
    the test hermetic."""
    import importlib
    ingest_mod = importlib.import_module("sunder.ingest")

    cfg = SunderConfig(
        curate=True,
        curation_goals={"purpose": "test"},
        anthropic_api_key="sk-test",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps([
        {
            "section_id": 0,
            "doc_id": "d",
            "decision": "KEEP",
            "reason": "ok",
            "original": "approved curated text body",
            "chunks": ["approved curated text body"],
        }
    ]))

    # If Claude got called we'd know curation ran again -- it must not.
    def boom(*a, **kw):
        raise AssertionError("curate_text must not run when report exists")

    import sunder.curator as curator_mod
    monkeypatch.setattr(curator_mod, "curate_text", boom)

    # Stub the embed/index stage so we don't need a real model.
    sentinel = SimpleNamespace(index=SimpleNamespace(ntotal=1))

    def fake_index(chunks, output_dir, config):
        # Must have received chunks built from the approved text.
        assert any("approved curated text body" in c.text for c in chunks)
        return sentinel

    monkeypatch.setattr(ingest_mod, "index_embeddings", fake_index)

    # Use flat chunking to avoid needing an LLM provider.
    cfg.chunking_mode = "flat"
    cfg.min_chunk_size = 1

    result = ingest(
        docs=["approved curated text body"],
        output_dir=str(tmp_path / "out"),
        config=cfg,
        curation_report_path=str(report_path),
    )

    assert result is sentinel
