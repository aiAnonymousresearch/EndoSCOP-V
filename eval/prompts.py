"""System prompt and per-turn message construction."""

from __future__ import annotations

_SYSTEM_PROMPT = """\
You are an expert gastroenterologist and endoscopist reviewing a {procedure_type} video. \
You are answering questions about the video based solely on your visual assessment of the \
endoscopic footage. Frame numbers are displayed in green font in the top-left corner of \
the video.

For each question, respond ONLY with a valid JSON object matching the provided schema. \
Do not include any explanation, reasoning, or text outside the JSON object."""


def build_system_prompt(procedure_type: str) -> str:
    """Build the system prompt for a given procedure type."""
    return _SYSTEM_PROMPT.format(procedure_type=procedure_type)


def build_turn_message(question: dict) -> str:
    """
    Build the user message text for a single conversation turn.

    Handles:
    - disease_reveal_prefix (turn 5 / disease transitions)
    - answer_reveal_prefix (conditional questions)
    - MCQ options (dict with letter keys)
    - Binary options (list of 2 values)
    """
    parts: list[str] = []

    # Some QA files emit `null` for reveal prefixes instead of omitting the key.
    if question.get("disease_reveal_prefix"):
        parts.append(question["disease_reveal_prefix"])
        parts.append("")

    if question.get("answer_reveal_prefix"):
        parts.append(question["answer_reveal_prefix"])
        parts.append("")

    parts.append(question["stem"])
    parts.append("")

    options = question["options"]
    if isinstance(options, dict):
        for letter, text in options.items():
            parts.append(f"{letter}. {text}")
    elif isinstance(options, list):
        parts.append("Options: " + " / ".join(str(o) for o in options))

    parts.append("")
    parts.append(f"Respond with JSON: {_describe_schema(question['answer_schema'])}")

    return "\n".join(parts)


def _describe_schema(schema: dict) -> str:
    """Human-readable description of the expected answer format."""
    props = schema.get("properties", {})
    answer_prop = props.get("answer", {})
    if answer_prop.get("type") == "array":
        items_enum = answer_prop.get("items", {}).get("enum", [])
        vals = ", ".join(str(v) for v in items_enum)
        return f'{{"answer": ["X", ...]}} where each element is one of: {vals}'
    else:
        enum = answer_prop.get("enum", [])
        vals = ", ".join(str(v) for v in enum)
        return f'{{"answer": "X"}} where X is one of: {vals}'
