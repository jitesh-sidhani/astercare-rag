from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Make project root importable when running:
# python scripts\evaluate.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import RAGPipeline  # noqa: E402


DEFAULT_EVAL_PATH = ROOT / "data" / "evaluation_set.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "evaluation_results.json"


def regex_search(pattern: str, text: str) -> bool:
    """Case-insensitive regex search with safe error handling."""
    try:
        return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None
    except re.error:
        return False


def check_required_patterns(
    answer: str,
    patterns: list[str],
) -> list[str]:
    """Return patterns that do not match the answer."""
    missing: list[str] = []

    for pattern in patterns:
        if not regex_search(pattern, answer):
            missing.append(pattern)

    return missing


def check_forbidden_patterns(
    answer: str,
    patterns: list[str],
) -> list[str]:
    """Return forbidden regex patterns that match the answer."""
    found: list[str] = []

    for pattern in patterns:
        if regex_search(pattern, answer):
            found.append(pattern)

    return found


def check_forbidden_values(
    answer: str,
    values: list[str],
) -> list[str]:
    """
    Check for sensitive values that must never be disclosed.

    This is intentionally a direct value check because restricted numeric
    values should never appear in the generated answer.
    """
    answer_lower = answer.lower()
    found: list[str] = []

    for value in values:
        if value.lower() in answer_lower:
            found.append(value)

    return found


def get_source_headings(result: dict[str, Any]) -> list[str]:
    headings: list[str] = []

    for source in result.get("sources", []):
        heading = source.get("heading")

        if heading:
            headings.append(str(heading))

    return headings


def check_required_sources(
    result: dict[str, Any],
    expected_headings: list[str],
) -> list[str]:
    """
    Return expected source headings that were not retrieved.

    Matching is case-insensitive and allows partial heading matches.
    """
    actual = [
        heading.lower().strip()
        for heading in get_source_headings(result)
    ]

    missing: list[str] = []

    for expected in expected_headings:
        expected_normalized = expected.lower().strip()

        matched = any(
            expected_normalized in actual_heading
            or actual_heading in expected_normalized
            for actual_heading in actual
        )

        if not matched:
            missing.append(expected)

    return missing


def check_refusal(answer: str) -> bool:
    """
    Check whether the answer gives a reasonable refusal / insufficiency
    response for restricted or out-of-scope questions.

    We intentionally accept multiple phrasings rather than requiring one
    exact sentence.
    """
    refusal_patterns = [
        r"(?i)does\s+not\s+provide\s+enough\s+information",
        r"(?i)does\s+not\s+provide\s+the\s+information",
        r"(?i)not\s+enough\s+information",
        r"(?i)insufficient\s+(information|context)",
        r"(?i)cannot\s+(provide|disclose|answer)",
        r"(?i)unable\s+to\s+(provide|answer|disclose)",
        r"(?i)not\s+available\s+from\s+the\s+provided\s+(context|handbook)",
        r"(?i)outside\s+the\s+scope",
        r"(?i)not\s+covered\s+by\s+the\s+handbook",
    ]

    return any(regex_search(pattern, answer) for pattern in refusal_patterns)


def evaluate_case(
    pipeline: RAGPipeline,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Run one evaluation case and calculate structured rule-based checks."""
    question = case["question"]

    try:
        result = pipeline.run(question)
        answer = result.get("answer", "") or ""

        required_patterns = case.get("required_patterns", [])
        forbidden_patterns = case.get("forbidden_patterns", [])
        forbidden_values = case.get("forbidden_values", [])
        required_sources = case.get("required_sources", [])
        must_refuse = bool(case.get("must_refuse", False))

        missing_required = check_required_patterns(
            answer,
            required_patterns,
        )

        forbidden_found = check_forbidden_patterns(
            answer,
            forbidden_patterns,
        )

        forbidden_values_found = check_forbidden_values(
            answer,
            forbidden_values,
        )

        missing_sources = check_required_sources(
            result,
            required_sources,
        )

        refusal_passed = True

        if must_refuse:
            refusal_passed = check_refusal(answer)

        passed = (
            not missing_required
            and not forbidden_found
            and not forbidden_values_found
            and not missing_sources
            and refusal_passed
        )

        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "question": question,
            "passed": passed,
            "answer": answer,
            "missing_required": missing_required,
            "forbidden_found": forbidden_found,
            "forbidden_values_found": forbidden_values_found,
            "missing_expected_sources": missing_sources,
            "refusal_check": (
                None if not must_refuse else refusal_passed
            ),
            "sources": result.get("sources", []),
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "question": question,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "answer": "",
            "missing_required": [],
            "forbidden_found": [],
            "forbidden_values_found": [],
            "missing_expected_sources": [],
            "refusal_check": None,
            "sources": [],
        }


def print_case_summary(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"

    print(f"[{status}] {result['id']} | {result['category']}")
    print(f"  Q: {result['question']}")

    if result.get("error"):
        print(f"  Error: {result['error']}")
        return

    if result["missing_required"]:
        print("  Missing required patterns:")
        for pattern in result["missing_required"]:
            print(f"    - {pattern}")

    if result["forbidden_found"]:
        print("  Forbidden claims found:")
        for pattern in result["forbidden_found"]:
            print(f"    - {pattern}")

    if result["forbidden_values_found"]:
        print(
            "  Forbidden values found: "
            + ", ".join(result["forbidden_values_found"])
        )

    if result["missing_expected_sources"]:
        print(
            "  Missing expected sources: "
            + ", ".join(result["missing_expected_sources"])
        )

    if result.get("refusal_check") is False:
        print("  Refusal check: FAIL")

    source_headings = get_source_headings(result)

    if source_headings:
        print("  Sources: " + " | ".join(source_headings))

    answer_preview = result.get("answer", "").replace("\n", " ")

    if len(answer_preview) > 220:
        answer_preview = answer_preview[:217] + "..."

    print(f"  Answer: {answer_preview}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the AsterCare RAG pipeline against "
            "a structured evaluation set."
        )
    )

    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to evaluation_set.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for detailed evaluation_results.json",
    )

    args = parser.parse_args()

    eval_path = args.eval_file

    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path

    output_path = args.output

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not eval_path.exists():
        print(f"Evaluation file not found: {eval_path}")
        return 1

    try:
        with eval_path.open("r", encoding="utf-8") as f:
            cases = json.load(f)

    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in evaluation file: {exc}")
        return 1

    if not isinstance(cases, list) or not cases:
        print("Evaluation file must contain a non-empty JSON list.")
        return 1

    print("Loading RAG pipeline...")
    pipeline = RAGPipeline()

    print(f"Running {len(cases)} evaluation cases...\n")

    results: list[dict[str, Any]] = []

    for case in cases:
        result = evaluate_case(pipeline, case)
        results.append(result)

        print_case_summary(result)
        print()

    total = len(results)
    passed = sum(
        1 for result in results
        if result["passed"]
    )
    failed = total - passed
    accuracy = passed / total if total else 0.0

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(accuracy, 4),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "summary": summary,
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total:     {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {failed}")
    print(f"Pass rate: {accuracy:.2%}")
    print(f"Results:   {output_path}")

    # Non-zero exit is still useful for CI / automated testing.
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())