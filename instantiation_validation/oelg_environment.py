"""Deterministic environment rules, compatible with environment_descrption.py.

Rules describe screening scenarios, not calibrated hazard probabilities. Relative
elevation is measured against the configured regional mean, never water depth.
"""
import json
import math
from pathlib import Path

DEFAULT_ENVIRONMENT_RULES = Path(__file__).parent / 'data/dictionary/oelg_environment_rules.json'
FACTOR_FIELDS = {
    'slope': ('avg_slope', 'slope'),
    'abs_elevation': ('avg_elevation', 'elevation'),
    'rel_elevation': ('avg_elevation', 'elevation'),
    'cape': ('cape',), 'moisture_convergence': ('moisture_convergence',),
    'wind_exposed': ('wind_exposed',), 'freeboard_m': ('freeboard_m',),
    'soil_moisture': ('soil_moisture',), 'air_temperature': ('air_temperature',),
    'population_density': ('population_density',), 'sanitation_index': ('sanitation_index',),
    'egress_routes': ('egress_routes',), 'evacuation_zone': ('evacuation_zone',),
    'supply_reserve_days': ('supply_reserve_days',),
    'accessible_supply_routes': ('accessible_supply_routes',),
}
UNITS = {'slope': '°', 'abs_elevation': 'm', 'rel_elevation': 'm',
         'cape': 'J/kg', 'moisture_convergence': 'kg/(m2*s)', 'wind_exposed': '0/1',
         'freeboard_m': 'm', 'soil_moisture': 'm3/m3', 'air_temperature': '°C',
         'population_density': 'persons/km2', 'sanitation_index': '0-1',
         'egress_routes': 'count', 'evacuation_zone': '0/1',
         'supply_reserve_days': 'day', 'accessible_supply_routes': 'count'}
TARGETS = {'Terrain', 'Slope_Body', 'Drainage_Area', 'Water_Body', 'Urban_Area'}


class MissingEnvironmentData(ValueError):
    """Unknown suitability due to absent/invalid input, not a negative result."""


def number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a numeric measurement')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Measurement must be finite')
    return result


def validate_spatial_rule(rule):
    if not isinstance(rule, dict) or rule.get('target_object') not in TARGETS:
        raise ValueError('Invalid environment target_object')
    if rule.get('logic') not in ('AND', 'OR') or not isinstance(rule.get('constraints'), list):
        raise ValueError('Environment logic must be AND/OR with constraints')
    for condition in rule['constraints']:
        if not isinstance(condition, dict) or condition.get('factor') not in FACTOR_FIELDS:
            raise ValueError('Unknown environment factor')
        if condition.get('unit') != UNITS[condition['factor']]:
            raise ValueError('Environment factor unit mismatch')
        if condition.get('operator') not in ('>', '<', '>=', '<=', '==', 'between'):
            raise ValueError('Invalid environment operator')
        value = condition.get('value')
        if condition['operator'] == 'between':
            if not isinstance(value, list) or len(value) != 2 or number(value[0]) > number(value[1]):
                raise ValueError('between requires ordered [minimum, maximum]')
        else:
            number(value)


def validate_environment_rule(rule):
    for key in ('phenomenon', 'description', 'version', 'basis', 'limitations'):
        if not isinstance(rule.get(key), str) or not rule[key].strip():
            raise ValueError(f'Environment rule lacks {key}')
    if rule.get('schema_version') != 1 or rule.get('quality') not in ('scenario_proxy', 'requires_additional_data'):
        raise ValueError('Invalid environment schema or quality')
    validate_spatial_rule(rule['spatial_rule'])
    environment = rule.get('environment')
    if environment is not None:
        if (not isinstance(environment, dict) or not isinstance(environment.get('geo_types'), list)
                or not environment['geo_types'] or any(not isinstance(t, str) or not t.strip()
                                                      for t in environment['geo_types'])):
            raise ValueError('Environment geometry requires geographic types')
        if number(environment.get('buffer_m')) < 0:
            raise ValueError('Environment buffer must be nonnegative')
        validate_spatial_rule(environment['spatial_rule'])
    if not rule['spatial_rule']['constraints'] and environment is None:
        raise ValueError('An environment rule must contain conditions or an environment geometry')
    if 'observed_root_bypass' in rule and type(rule['observed_root_bypass']) is not bool:
        raise ValueError('observed_root_bypass must be boolean')
    return rule


def load_environment_catalog(path=DEFAULT_ENVIRONMENT_RULES):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('schema_version') != 1 or not isinstance(data.get('rules'), list) or not data['rules']:
        raise ValueError('Invalid environment catalog')
    names = set()
    for rule in data['rules']:
        validate_environment_rule(rule)
        if rule['phenomenon'] in names:
            raise ValueError('Duplicate phenomenon environment rule')
        names.add(rule['phenomenon'])
    return data


def evaluate_spatial_rule(properties, rule, regional_mean_elevation_m):
    """Three-valued evaluation: True/False or MissingEnvironmentData.

    A true OR branch, or a false AND branch, determines the result even if a
    different branch is unknown. Never substitute 0 for absent measurements.
    Validation is done when rules are loaded, rather than once per feature.
    """
    results, missing = [], []
    for condition in rule['constraints']:
        factor = condition['factor']
        fields = FACTOR_FIELDS[factor]
        raw = next((properties[k] for k in fields if properties.get(k) is not None), None)
        try:
            actual = number(raw)
            if factor == 'slope' and not 0 <= actual <= 90:
                raise ValueError('Slope outside 0..90')
            if factor in ('soil_moisture', 'sanitation_index') and not 0 <= actual <= 1:
                raise ValueError('Ratio outside 0..1')
            if factor in ('wind_exposed', 'evacuation_zone') and actual not in (0, 1):
                raise ValueError('Flag outside 0/1')
            if factor == 'rel_elevation':
                actual -= number(regional_mean_elevation_m)
        except (TypeError, ValueError):
            missing.append(factor)
            continue
        value, operator = condition['value'], condition['operator']
        if operator == 'between':
            result = number(value[0]) <= actual <= number(value[1])
        else:
            expected = number(value)
            result = {'>': actual > expected, '<': actual < expected,
                      '>=': actual >= expected, '<=': actual <= expected,
                      '==': actual == expected}[operator]
        results.append(result)
    if rule['logic'] == 'OR' and any(results):
        return True
    if rule['logic'] == 'AND' and not all(results):
        return False
    if missing:
        raise MissingEnvironmentData(', '.join(sorted(set(missing))))
    return any(results) if rule['logic'] == 'OR' and results else all(results)
