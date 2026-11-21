"""Appendix E causal priors shared by graph construction and inference.

A permitted pair is a candidate, not proof of event-level causation. Preliminary
same-type rules concern distinct events and spatial objects, never self-loops.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


IMMEDIATE = 'I'
ALTERING = 'A'
RELATION_LABELS = {IMMEDIATE: 'Immediate', ALTERING: 'Altering'}
RULES_PATH = Path(__file__).resolve().parents[1] / 'data/dictionary/causal_relation_rules.json'
with RULES_PATH.open(encoding='utf-8') as stream:
    _SOURCE = json.load(stream)
RELATION_CONFIDENCE_MULTIPLIERS = _SOURCE['coefficient_by_code']
_ACTIVE_RULES = {(row['source'], row['target']): row for row in _SOURCE['records']
                 if row['review_status'] != 'Remove'}
_PHENOMENA = {name for pair in _ACTIVE_RULES for name in pair}
_CANONICAL = {' '.join(name.split()).casefold(): name for name in _PHENOMENA}
CAUSAL_RELATION_MATRIX: Dict[str, Dict[str, str]] = {}
for (_source, _target), _rule in _ACTIVE_RULES.items():
    CAUSAL_RELATION_MATRIX.setdefault(_source, {})[_target] = _rule['code']

# Compatibility imports use the Appendix E codes; new output always uses I/A.
TRIGGER = IMMEDIATE
FACILITATE = ALTERING


def canonical_impact_type(value: str) -> Optional[str]:
    """Resolve a phenomenon by case and whitespace without inferring synonyms."""
    return _CANONICAL.get(' '.join(str(value or '').split()).casefold())


def get_relation_code(primary_impact_type: str, secondary_impact_type: str) -> Optional[str]:
    """Return I/A for an active Appendix E pair, otherwise None."""
    source = canonical_impact_type(primary_impact_type)
    target = canonical_impact_type(secondary_impact_type)
    return CAUSAL_RELATION_MATRIX.get(source, {}).get(target)


def get_relation_metadata(primary_impact_type: str, secondary_impact_type: str) -> Optional[dict]:
    """Return graph properties for a candidate pair and its original review status."""
    pair = (canonical_impact_type(primary_impact_type), canonical_impact_type(secondary_impact_type))
    rule = _ACTIVE_RULES.get(pair)
    if rule is None:
        return None
    label = RELATION_LABELS[rule['code']]
    return dict(relation_type=label, relation_name=label, affectType=label,
                matrix_relation_code=rule['code'], matrix_review_status=rule['review_status'])


def get_allowed_secondary_impact_types(primary_impact_type: str) -> Set[str]:
    """Return candidate secondary phenomenon types allowed by Appendix E."""
    return set(CAUSAL_RELATION_MATRIX.get(canonical_impact_type(primary_impact_type), {}))


def calculate_relation_confidence(base_confidence: float, relation_code: str) -> float:
    """Apply the paper coefficient once: Immediate=1 and Altering=0.65."""
    code = {'T': IMMEDIATE, 'F': ALTERING}.get(relation_code, relation_code)
    if code not in RELATION_CONFIDENCE_MULTIPLIERS:
        raise ValueError(f'Unsupported causal relation code: {relation_code!r}')
    return max(0.0, float(base_confidence or 0.0)) * RELATION_CONFIDENCE_MULTIPLIERS[code]


def active_relation_counts() -> Tuple[int, int, int]:
    """Return (Immediate, Altering, total) counts for active Appendix E rules."""
    codes = [row['code'] for row in _ACTIVE_RULES.values()]
    return codes.count(IMMEDIATE), codes.count(ALTERING), len(codes)
