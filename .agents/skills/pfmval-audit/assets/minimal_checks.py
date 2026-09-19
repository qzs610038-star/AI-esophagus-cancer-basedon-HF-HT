"""Portable audit helpers; Python 3.10+, standard library, no I/O.

Compare declared expectations with independently obtained runtime observations.
Results describe evidence, not training authorization or scientific optimality.
"""

from collections import Counter
import math


def _duplicates(values):
    return {key: count for key, count in Counter(values).items() if count > 1}


def check_identity_order(expected, actual, label="identity"):
    """Compare sequences of hashable identity keys (or pathway names)."""
    expected, actual = list(expected), list(actual)
    expected_set, actual_set = set(expected), set(actual)
    missing = list(dict.fromkeys(key for key in expected if key not in actual_set))
    extra = list(dict.fromkeys(key for key in actual if key not in expected_set))
    duplicates_expected = _duplicates(expected)
    duplicates_actual = _duplicates(actual)
    mismatch_count = 0
    examples = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            mismatch_count += 1
            if len(examples) < 5:
                examples.append({"index": index, "expected": left, "actual": right})
    mismatch_count += abs(len(expected) - len(actual))
    different = bool(missing or extra or duplicates_expected or duplicates_actual or mismatch_count)
    return {
        "status": "FAIL" if different else "PASS" if expected else "UNVERIFIED",
        "label": label,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extra": extra,
        "duplicates_expected": duplicates_expected,
        "duplicates_actual": duplicates_actual,
        "order_mismatch_count": mismatch_count,
        "order_mismatch_examples": examples,
    }


def compare_contract(expected, observed, tolerances=None):
    """Compare named plain Python values; absent observed keys are unverified.

    Extra observed fields are ignored. Optional tolerances map a field to
    (relative, absolute) nonnegative finite tolerances for numeric scalars.
    None is a legitimate value, not a missing-evidence marker.
    """
    tolerances = tolerances or {}
    if set(tolerances) - set(expected):
        raise ValueError("Tolerance fields must belong to expected declarations")
    for relative, absolute in tolerances.values():
        if not all(math.isfinite(v) and v >= 0 for v in (relative, absolute)):
            raise ValueError("Tolerances must be finite and nonnegative")
    matched, unverified, mismatches = [], [], []
    for name, wanted in expected.items():
        if name not in observed:
            unverified.append(name)
            continue
        got = observed[name]
        # bool is a subclass of int; True must not silently match batch_size=1.
        is_number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if is_number(wanted) and is_number(got):
            relative, absolute = tolerances.get(name, (0, 0))
            equal = all(not isinstance(v, float) or math.isfinite(v) for v in (wanted, got))
            # Exact mode must not round large integer seeds/counts to floats.
            equal = equal and (wanted == got if relative == absolute == 0 else
                               math.isclose(wanted, got, rel_tol=relative, abs_tol=absolute))
        elif isinstance(wanted, bool) or isinstance(got, bool):
            equal = type(wanted) is type(got) and wanted == got
        else:
            equal = wanted == got
        if equal:
            matched.append(name)
        else:
            mismatches.append({"field": name, "expected": wanted, "observed": got})
    return {
        "status": "FAIL" if mismatches else "UNVERIFIED" if unverified or not expected else "PASS",
        "matched": matched,
        "unverified": unverified,
        "mismatches": mismatches,
        "expectations_declared": bool(expected),
    }


def check_result_coverage(expected, actual, batch_status=None):
    """Check declared (model, seed, split) scope, not a fixed model/seed count.

    actual records contain model, seed, split, status. Only 'succeeded' counts
    as success; callers explicitly translate their own status vocabulary.
    A missing status is unknown. batch_status=None checks only declared scope.
    Partial results are WARN, not automatically an implementation defect.
    """
    expected = [tuple(key) for key in expected]
    if any(len(key) != 3 for key in expected):
        raise ValueError("Expected keys must be (model, seed, split) triples")
    rows = list(actual)
    actual_keys = [(row["model"], row["seed"], row["split"]) for row in rows]
    expected_set, actual_set = set(expected), set(actual_keys)
    missing = list(dict.fromkeys(key for key in expected if key not in actual_set))
    extra = list(dict.fromkeys(key for key in actual_keys if key not in expected_set))
    duplicate_expected = _duplicates(expected)
    duplicate_actual = _duplicates(actual_keys)
    unsuccessful = [
        {"key": key, "status": row.get("status", "unknown")}
        for key, row in zip(actual_keys, rows)
        if row.get("status") != "succeeded"
    ]
    complete = bool(expected) and not (
        missing or extra or duplicate_expected or duplicate_actual or unsuccessful
        or batch_status not in (None, "succeeded")
    )
    return {
        "status": "UNVERIFIED" if not expected else "PASS" if complete else "WARN",
        "complete": complete,
        "expected_count": len(expected),
        "actual_count": len(actual_keys),
        "successful_count": sum(row.get("status") == "succeeded" for row in rows),
        "missing": missing,
        "extra": extra,
        "duplicates_expected": duplicate_expected,
        "duplicates_actual": duplicate_actual,
        "unsuccessful": unsuccessful,
        "batch_status": batch_status,
        "batch_status_checked": batch_status is not None,
    }
