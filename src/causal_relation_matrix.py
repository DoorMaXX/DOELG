"""Class-level causal relation constraints for multi-hazard instantiation.

This module is the code representation of the active (non-Remove) entries in
``Predefined_Causal_Temporal_Relation_Matrix.xlsx`` (Candidate Relations).
An edge is intentionally a *candidate* relation: it must still pass the
event-level spatial, environmental, and evidence checks implemented by the
instantiation workflow.

A diagonal entry represents same-type propagation between distinct event
instances and distinct spatial objects.  It never authorizes a causal
self-loop from an event node back to itself; callers must reject identical
source and target event instances before creating an edge.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple


TRIGGER = "T"
FACILITATE = "F"

# These are explicit modelling parameters, not elapsed-time values.  Trigger
# preserves the evidence confidence of a direct edge; Facilitate discounts it
# because the secondary event additionally depends on thresholds, accumulation,
# or contextual conditions.  Calibrate the Facilitate value with instantiation_validation data
# when such data become available.
RELATION_CONFIDENCE_MULTIPLIERS: Dict[str, float] = {
    TRIGGER: 1.00,
    FACILITATE: 0.65,
}


def _relations(trigger: Iterable[str] = (), facilitate: Iterable[str] = ()) -> Dict[str, str]:
    """Build a compact secondary-impact → relation-code mapping."""
    return {
        **{impact_type: TRIGGER for impact_type in trigger},
        **{impact_type: FACILITATE for impact_type in facilitate},
    }


# Keep this mapping synchronized with the active (non-Remove) records in the
# Candidate Relations worksheet.  The matrix contains 194 active relations:
# 87 Triggering and 107 Facilitation.
CAUSAL_RELATION_MATRIX: Dict[str, Dict[str, str]] = {
    "Atmospheric Convection": _relations(
        trigger=("Atmospheric Convection", "Extreme Precipitation", "Wind Intensification"),
    ),
    "Extreme Precipitation": _relations(
        trigger=("Water Level Rise", "Rapid Flow", "Waterlogging", "Soil Erosion"),
        facilitate=("Landslide", "Debris Flow", "Subsidence", "Water Pollution", "Obstructed Drainage"),
    ),
    "Wind Intensification": _relations(
        trigger=("Structural Fracture", "Energy Disruption", "Communication Disruption"),
        facilitate=("Structural Collapse", "Traffic Disruption", "Blockage"),
    ),
    "Water Level Rise": _relations(
        trigger=("Water Level Rise", "Water Overflow", "Submergence"),
        facilitate=("Water Backflow", "Dike Breach", "Waterlogging"),
    ),
    "Rapid Flow": _relations(
        trigger=("Rapid Flow", "Soil Erosion"),
        facilitate=("Dike Breach", "Structural Fracture", "Structural Collapse", "Personnel Stranding", "Casualties"),
    ),
    "Water Overflow": _relations(
        trigger=("Waterlogging", "Submergence", "Traffic Disruption", "Personnel Stranding", "Evacuation"),
        facilitate=("Operation Stagnation", "Energy Disruption", "Communication Disruption", "Structural Fracture", "Structural Collapse", "Water Pollution", "Casualties", "Production Loss"),
    ),
    "Flood Discharge/Diversion": _relations(
        trigger=("Flood Discharge/Diversion", "Water Level Rise", "Rapid Flow"),
        facilitate=("Water Overflow", "Submergence"),
    ),
    "Backwater Effect": _relations(
        trigger=("Backwater Effect", "Water Level Rise", "Water Backflow"),
        facilitate=("Waterlogging",),
    ),
    "Water Backflow": _relations(
        trigger=("Water Backflow", "Waterlogging"),
        facilitate=("Water Pollution",),
    ),
    "Sediment Siltation": _relations(
        trigger=("Blockage", "Obstructed Drainage"),
        facilitate=("Waterlogging", "Water Backflow"),
    ),
    "Soil Erosion": _relations(
        trigger=("Sediment Siltation",),
        facilitate=("Soil Erosion", "Landslide", "Debris Flow", "Dike Breach"),
    ),
    "Dike Breach": _relations(
        trigger=("Dike Breach", "Rapid Flow", "Water Overflow", "Waterlogging", "Submergence", "Traffic Disruption", "Personnel Stranding", "Evacuation"),
        facilitate=("Casualties", "Structural Collapse", "Water Pollution"),
    ),
    "Waterlogging": _relations(
        trigger=("Waterlogging", "Traffic Disruption", "Operation Stagnation", "Personnel Stranding", "Evacuation"),
        facilitate=("Submergence", "Energy Disruption", "Communication Disruption", "Casualties", "Water Pollution", "Production Loss"),
    ),
    "Submergence": _relations(
        trigger=("Traffic Disruption", "Energy Disruption", "Operation Stagnation", "Personnel Stranding", "Evacuation"),
        facilitate=("Structural Collapse", "Structural Fracture", "Water Pollution", "Communication Disruption", "Casualties", "Supply Crisis", "Production Loss"),
    ),
    "Water Spillage": _relations(
        trigger=("Waterlogging",),
        facilitate=("Submergence", "Water Pollution", "Traffic Disruption"),
    ),
    "Landslide": _relations(
        trigger=("Blockage", "Structural Collapse", "Structural Fracture", "Traffic Disruption", "Evacuation"),
        facilitate=("Landslide", "Debris Flow", "Personnel Stranding", "Casualties", "Production Loss"),
    ),
    "Debris Flow": _relations(
        trigger=("Blockage", "Structural Collapse", "Traffic Disruption", "Evacuation"),
        facilitate=("Personnel Stranding", "Casualties", "Water Pollution"),
    ),
    "Blockage": _relations(
        trigger=("Backwater Effect", "Water Level Rise", "Waterlogging", "Obstructed Drainage"),
        facilitate=("Water Overflow", "Traffic Disruption"),
    ),
    "Subsidence": _relations(
        trigger=("Structural Fracture", "Traffic Disruption"),
        facilitate=("Subsidence", "Waterlogging", "Obstructed Drainage", "Structural Collapse", "Operation Stagnation", "Evacuation", "Production Loss"),
    ),
    "Structural Collapse": _relations(
        trigger=("Structural Collapse", "Traffic Disruption", "Operation Stagnation", "Casualties", "Personnel Stranding", "Evacuation", "Production Loss"),
        facilitate=("Energy Disruption", "Communication Disruption"),
    ),
    "Structural Fracture": _relations(
        trigger=("Traffic Disruption",),
        facilitate=("Structural Fracture", "Structural Collapse", "Energy Disruption", "Communication Disruption", "Operation Stagnation", "Casualties", "Evacuation", "Production Loss"),
    ),
    "Water Pollution": _relations(
        trigger=("Water Pollution",),
        facilitate=("Epidemic Spread", "Supply Crisis", "Casualties"),
    ),
    "Plant Pests and Diseases": _relations(
        facilitate=("Plant Pests and Diseases", "Supply Crisis", "Production Loss"),
    ),
    "Epidemic Spread": _relations(
        facilitate=("Epidemic Spread", "Casualties", "Operation Stagnation", "Supply Crisis", "Production Loss"),
    ),
    "Traffic Disruption": _relations(
        trigger=("Traffic Disruption", "Personnel Stranding"),
        facilitate=("Evacuation", "Supply Crisis", "Production Loss", "Operation Stagnation"),
    ),
    "Energy Disruption": _relations(
        trigger=("Energy Disruption", "Communication Disruption", "Operation Stagnation", "Production Loss"),
        facilitate=("Supply Crisis",),
    ),
    "Communication Disruption": _relations(
        trigger=("Communication Disruption", "Operation Stagnation"),
        facilitate=("Evacuation", "Personnel Stranding"),
    ),
    "Operation Stagnation": _relations(
        trigger=("Supply Crisis", "Production Loss"),
        facilitate=("Operation Stagnation",),
    ),
    "Obstructed Drainage": _relations(
        trigger=("Obstructed Drainage", "Waterlogging"),
        facilitate=("Water Level Rise", "Submergence"),
    ),
    "Personnel Stranding": _relations(
        trigger=("Evacuation",),
        facilitate=("Casualties",),
    ),
    "Evacuation": _relations(
        facilitate=("Supply Crisis", "Operation Stagnation"),
    ),
    "Supply Crisis": _relations(
        facilitate=("Supply Crisis", "Casualties", "Production Loss"),
    ),
}


def get_relation_code(primary_impact_type: str, secondary_impact_type: str) -> Optional[str]:
    """Return ``T``/``F`` for an active matrix edge, otherwise ``None``."""
    return CAUSAL_RELATION_MATRIX.get(primary_impact_type, {}).get(secondary_impact_type)


def get_allowed_secondary_impact_types(primary_impact_type: str) -> Set[str]:
    """Return the matrix-authorized secondary impact types for one primary type."""
    return set(CAUSAL_RELATION_MATRIX.get(primary_impact_type, {}))


def calculate_relation_confidence(base_confidence: float, relation_code: str) -> float:
    """Apply the qualitative T/F confidence modifier without introducing time bins."""
    if relation_code not in RELATION_CONFIDENCE_MULTIPLIERS:
        raise ValueError(f"Unsupported causal relation code: {relation_code!r}")
    return max(0.0, float(base_confidence or 0.0)) * RELATION_CONFIDENCE_MULTIPLIERS[relation_code]


def active_relation_counts() -> Tuple[int, int, int]:
    """Return (Triggering, Facilitation, total) for a quick configuration check."""
    trigger_count = sum(code == TRIGGER for targets in CAUSAL_RELATION_MATRIX.values() for code in targets.values())
    facilitate_count = sum(code == FACILITATE for targets in CAUSAL_RELATION_MATRIX.values() for code in targets.values())
    return trigger_count, facilitate_count, trigger_count + facilitate_count
