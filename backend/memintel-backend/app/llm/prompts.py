"""
app/llm/prompts.py
────────────────────────────────────────────────────────────────────────────────
LLM prompt building helpers.

build_context_prefix(context) builds a structured plain-text prefix from an
ApplicationContext for injection at the START of the LLM system prompt, before
all other instructions.

When context is None the function returns an empty string — callers proceed
normally with no prefix and no warning in the prompt itself.
"""
from __future__ import annotations

from app.models.context import ApplicationContext


def build_context_prefix(context: ApplicationContext | None) -> str:
    """
    Build a structured plain-text context prefix from an ApplicationContext.

    Returns empty string if context is None.

    Format (sections omitted when their data is absent):
      === APPLICATION CONTEXT ===
      Domain: <description>

      Entities: <name (description), ...>

      Decision types: <decision1, ...>

      Behavioural constraints:
      - Data cadence: <cadence>
      - Meaningful activity windows: <min> to <max>    (if meaningful_windows set)
      - Regulatory environment: <reg1, ...>            (if regulatory non-empty)

      Semantic definitions:                            (omitted if empty)
      - <term>: <definition>
      ...

      Calibration sensitivity:                         (omitted if calibration_bias is None)
      - False negative cost: <cost>
      - False positive cost: <cost>
      - Bias direction: <direction>

      === END APPLICATION CONTEXT ===
    """
    if context is None:
        return ""

    lines: list[str] = ["=== APPLICATION CONTEXT ==="]
    lines.append(f"Domain: {context.domain.description}")

    if context.domain.entities:
        entity_str = ", ".join(
            f"{e.name} ({e.description})" for e in context.domain.entities
        )
        lines.append(f"\nEntities: {entity_str}")

    if context.domain.decisions:
        lines.append(f"\nDecision types: {', '.join(context.domain.decisions)}")

    # Behavioural constraints — always present (cadence always has a default)
    lines.append("\nBehavioural constraints:")
    lines.append(f"- Data cadence: {context.behavioural.data_cadence}")

    if context.behavioural.meaningful_windows:
        mw = context.behavioural.meaningful_windows
        min_val = mw.get("min", "")
        max_val = mw.get("max", "")
        lines.append(f"- Meaningful activity windows: {min_val} to {max_val}")

    if context.behavioural.regulatory:
        lines.append(
            f"- Regulatory environment: {', '.join(context.behavioural.regulatory)}"
        )

    # Semantic definitions — omit section if empty
    if context.semantic_hints:
        lines.append("\nSemantic definitions:")
        for hint in context.semantic_hints:
            lines.append(f"- {hint.term}: {hint.definition}")

    # Calibration sensitivity — omit section if calibration_bias is None
    if context.calibration_bias is not None:
        bias = context.calibration_bias
        lines.append("\nCalibration sensitivity:")
        lines.append(f"- False negative cost: {bias.false_negative_cost}")
        lines.append(f"- False positive cost: {bias.false_positive_cost}")
        lines.append(f"- Bias direction: {bias.bias_direction}")

    lines.append("\n=== END APPLICATION CONTEXT ===")
    return "\n".join(lines) + "\n"
