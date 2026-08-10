"""Asks the evolver LLM to revise a bundle, applying exactly one validated change."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dr_cik.llm import JsonExtractionError, LLMClient, parse_json_object

from ..agents.common import extract_reasoning
from ..bundles import next_version, save_bundle
from ..harness.trace import emit_llm_response
from ..models import Bundle, BundleTriple
from .evaluate import TaskResult

logger = logging.getLogger(__name__)

EVOLVER_SYSTEM = (
    "You improve the configuration of an automated forecasting agent. You are shown the agent's "
    "current definition and concrete cases where it scored badly. Propose ONE targeted change that "
    "would plausibly fix the observed failures.\n\n"
    "You may change EITHER the system prompt OR exactly one code template -- never both, and never "
    "more than one template. Small, interpretable, well-motivated edits beat sweeping rewrites. "
    "Do not restate the failures back; change the definition so they stop happening."
)

MUTATION_SCHEMA = (
    'Respond with exactly one JSON object: {"change_type": "system_prompt" | "add_code_template" | '
    '"edit_code_template" | "remove_code_template", "target_template_name": "<name or null>", '
    '"system_prompt": "<full replacement text or null>", "code_template": "<full python source or null>", '
    '"changelog": "<one line, at most 200 characters>"}'
)

CHANGE_TYPES = ("system_prompt", "add_code_template", "edit_code_template", "remove_code_template")
_MAX_TRACE_CHARS = 1200
_MAX_CHANGELOG_CHARS = 200


class MutationError(ValueError):
    """Raised when an evolver response cannot be turned into exactly one legal change."""


def _render_trace(result: TaskResult) -> str:
    """Render one failing task compactly enough that several fit in a prompt."""
    body = json.dumps(result.trace, ensure_ascii=False, default=str)
    if len(body) > _MAX_TRACE_CHARS:
        body = body[:_MAX_TRACE_CHARS] + "...(truncated)"
    return f"- task {result.task_id} scored {result.score:.4f}\n  {body}"


def build_mutation_prompt(bundle: Bundle, worst_traces: list[TaskResult]) -> str:
    """Assemble the evolver prompt: the current definition plus concrete failures."""
    templates = json.dumps(bundle.code_templates, indent=2) if bundle.code_templates else "(none)"
    failures = "\n".join(_render_trace(result) for result in worst_traces) or "(no failing traces supplied)"
    return (
        f"Agent role: {bundle.agent}\n"
        f"Current version: {bundle.version}\n\n"
        f"Current system prompt:\n\"\"\"\n{bundle.system_prompt}\n\"\"\"\n\n"
        f"Current code templates:\n{templates}\n\n"
        f"Worst-scoring tasks this generation (higher score is better):\n{failures}\n\n"
        f"{MUTATION_SCHEMA}"
    )


def apply_change(bundle: Bundle, parsed: dict, version: str) -> Bundle:
    """Apply exactly one validated change to a copy of the parent bundle.

    Never builds a bundle from the response wholesale: version/parent/agent/hyperparameters are
    always carried over from the parent, so a misbehaving evolver cannot rewrite bookkeeping.
    """
    change_type = parsed.get("change_type")
    if change_type not in CHANGE_TYPES:
        raise MutationError(f"unknown change_type {change_type!r}")
    changelog = str(parsed.get("changelog", "")).strip()[:_MAX_CHANGELOG_CHARS]

    child = replace(
        bundle,
        bundle_id=f"{bundle.agent}/{version}",
        version=version,
        parent=bundle.version,
        notes_from_evolver=changelog,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    if change_type == "system_prompt":
        prompt = parsed.get("system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MutationError("system_prompt change carried no replacement text")
        return replace(child, system_prompt=prompt.strip())

    name = parsed.get("target_template_name")
    if not isinstance(name, str) or not name.strip():
        raise MutationError(f"{change_type} requires target_template_name")
    templates = dict(bundle.code_templates)

    if change_type == "remove_code_template":
        if name not in templates:
            raise MutationError(f"cannot remove absent template {name!r}")
        templates.pop(name)
        return replace(child, code_templates=templates)

    source = parsed.get("code_template")
    if not isinstance(source, str) or not source.strip():
        raise MutationError(f"{change_type} carried no code_template source")
    if change_type == "edit_code_template" and name not in templates:
        raise MutationError(f"cannot edit absent template {name!r}")
    if change_type == "add_code_template" and name in templates:
        raise MutationError(f"template {name!r} already exists; use edit_code_template")
    templates[name] = source
    return replace(child, code_templates=templates)


def mutate(bundle: Bundle, worst_traces: list[TaskResult], evolver_llm: LLMClient, bundles_dir: str | Path) -> Bundle:
    """Produce one child bundle; on any failure return an unchanged child so the run continues."""
    version = next_version(bundles_dir, bundle.agent)
    prompt = build_mutation_prompt(bundle, worst_traces)
    response = evolver_llm.complete(
        system=EVOLVER_SYSTEM, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_output_tokens=2048
    )
    reasoning, answer = extract_reasoning(response.text)
    emit_llm_response(bundle.bundle_id, "evolver", answer, reasoning, model_id=getattr(evolver_llm, "model_id", "?"))

    try:
        child = apply_change(bundle, parse_json_object(answer), version)
    except (JsonExtractionError, MutationError) as exc:
        logger.warning("mutate[%s]: %s; emitting an unchanged child", bundle.bundle_id, exc)
        child = replace(
            bundle,
            bundle_id=f"{bundle.agent}/{version}",
            version=version,
            parent=bundle.version,
            notes_from_evolver=f"mutation failed ({exc}), unchanged",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    save_bundle(child, bundles_dir)
    logger.info("mutate[%s] -> %s: %s", bundle.bundle_id, child.bundle_id, child.notes_from_evolver)
    return child


def mutate_triple(
    triple: BundleTriple,
    worst_traces: list[TaskResult],
    evolver_llm: LLMClient,
    bundles_dir: str | Path,
    rng: random.Random | None = None,
    decision_weight: float = 0.7,
) -> BundleTriple:
    """Mutate one slot of a triple, mostly the decision bundle, per the system-loop design."""
    generator = rng or random.Random()
    draw = generator.random()
    if draw < decision_weight:
        return replace(triple, decision=mutate(triple.decision, worst_traces, evolver_llm, bundles_dir))
    if draw < decision_weight + (1 - decision_weight) / 2:
        return replace(triple, coding=mutate(triple.coding, worst_traces, evolver_llm, bundles_dir))
    return replace(triple, retrieval=mutate(triple.retrieval, worst_traces, evolver_llm, bundles_dir))
