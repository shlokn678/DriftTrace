"""Privacy / PII deny-list checks (FR-15.5, NFR-9).

Checks that no field on the declared PII deny-list appears in a payload, event, or
record. Used to guard logs, artifacts, and event payloads. Deterministic and pure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from drifttrace.governance.config import GovernanceConfig


@dataclass
class PrivacyResult:
    """Result of a privacy check over one or more records."""

    passed: bool
    deny_list: list[str]
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_keys(obj: Any, keys: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            _collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, keys)


def check_payload(payload: Any, config: GovernanceConfig) -> PrivacyResult:
    """Fail if any deny-listed field name appears anywhere in ``payload``."""
    present: set[str] = set()
    _collect_keys(payload, present)
    deny = {d.lower() for d in config.pii_deny_list}
    violations = sorted(present & deny)
    return PrivacyResult(
        passed=len(violations) == 0,
        deny_list=list(config.pii_deny_list),
        violations=violations,
    )


def check_records(records: list[dict], config: GovernanceConfig) -> PrivacyResult:
    """Check a list of records (e.g. prediction events) against the deny-list."""
    all_violations: set[str] = set()
    for rec in records:
        result = check_payload(rec, config)
        all_violations.update(result.violations)
    return PrivacyResult(
        passed=len(all_violations) == 0,
        deny_list=list(config.pii_deny_list),
        violations=sorted(all_violations),
    )
