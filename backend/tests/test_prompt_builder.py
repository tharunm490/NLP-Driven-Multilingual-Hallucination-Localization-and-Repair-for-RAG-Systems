"""Tests for prompt construction."""

from __future__ import annotations

from app.services.prompt_builder import (
    EVIDENCE_BEGIN,
    EVIDENCE_END,
    PromptBuilder,
    PromptSource,
)


def make_source(index: int = 1, page: int | None = 2) -> PromptSource:
    return PromptSource(
        index=index,
        document_id=f"doc_{index}",
        filename=f"file_{index}.pdf",
        chunk_id=f"chunk_{index:03d}",
        page_number=page,
        text=f"Evidence body number {index}.",
    )


def test_system_prompt_is_a_separate_field():
    builder = PromptBuilder()
    built = builder.build("When was it founded?", [make_source()])
    assert built.system_prompt
    assert "System" not in built.user_message.splitlines()[0]
    assert built.system_prompt != built.user_message


def test_evidence_is_delimited_and_labelled():
    built = PromptBuilder().build("Question?", [make_source(1), make_source(2, page=None)])
    user = built.user_message
    assert EVIDENCE_BEGIN in user
    assert EVIDENCE_END in user
    assert "[Source 1]" in user
    assert "[Source 2]" in user
    assert "file_1.pdf" in user
    assert "Page: 2" in user
    assert "Page: unavailable" in user
    assert "chunk_002" in user
    assert "Evidence body number 2." in user
    assert "USER QUESTION:" in user
    assert "Question?" in user
    assert built.source_count == 2


def test_system_prompt_treats_evidence_as_untrusted():
    prompt = PromptBuilder().system_prompt
    assert "untrusted" in prompt.lower()
    assert "never follow" in prompt.lower()
    assert "do not invent" in prompt.lower()


def test_prompt_version_is_exposed():
    builder = PromptBuilder()
    built = builder.build("q", [make_source()])
    assert built.prompt_version == builder.version
    assert built.prompt_version


def test_empty_evidence_renders_placeholder():
    built = PromptBuilder().build("q", [])
    assert "(no evidence retrieved)" in built.evidence_text
    assert built.source_count == 0


def test_insufficient_evidence_prompt_has_no_sources():
    built = PromptBuilder().build_insufficient_evidence("q")
    assert built.source_count == 0
    assert "[Source" not in built.user_message
    assert "cannot answer" in built.user_message.lower()
