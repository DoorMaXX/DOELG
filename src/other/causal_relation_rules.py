"""Validate event-level CAUSE edges against Appendix E before graph ingestion."""

from src.causal_relation_matrix import canonical_impact_type, get_relation_metadata
from src.graph_language import english_term


def event_impact_type(event):
    """Read an event's typed impact argument without using its raw surface name."""
    impacts = event.impact_argument.all()
    if len(impacts) != 1:
        return None
    return canonical_impact_type(english_term(impacts[0].impact_type))


def classify_event_cause(source, target):
    """Classify a candidate and reject self-loops or same-object diagonal relations."""
    if source is target or source.node_id == target.node_id:
        return None
    source_type, target_type = event_impact_type(source), event_impact_type(target)
    metadata = get_relation_metadata(source_type, target_type)
    if metadata is None:
        return None
    if source_type == target_type:
        source_geos, target_geos = source.geo_argument.all(), target.geo_argument.all()
        if len(source_geos) != 1 or len(target_geos) != 1:
            return None
        first, second = source_geos[0], target_geos[0]
        if first.node_id == second.node_id or not first.geo_object_name or not second.geo_object_name:
            return None
        if first.geo_object_name.strip().casefold() == second.geo_object_name.strip().casefold():
            return None
    return metadata


def enforce_existing_causal_rules(event_nodes):
    """Filter existing extracted CAUSE edges on reruns, retaining evidence and counts."""
    removed = updated = 0
    for source in event_nodes:
        visited = set()
        for target in list(source.cause.all()):
            if target.node_id in visited:
                continue
            visited.add(target.node_id)
            metadata = classify_event_cause(source, target)
            relations = source.cause.all_relationships(target)
            if metadata is None:
                removed += len(relations)
                source.cause.disconnect(target)
                print(f'[Causal rule removed] {source.event_description} -> {target.event_description}')
                continue
            for relation in relations:
                if any(getattr(relation, key, None) != value for key, value in metadata.items()):
                    for key, value in metadata.items():
                        setattr(relation, key, value)
                    relation.save()
                    updated += 1
    return dict(removed=removed, updated=updated)
