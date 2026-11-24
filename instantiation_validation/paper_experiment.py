import csv
from pathlib import Path

import distance_fields as fields
from compare_importance_fields import make_comparison, selected_factor_fields
from evaluate_importance_fields import canonical_point_set, run as evaluate_fields

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('paper_experiment_config.json')


def load_profile(config_path=None):
    """Load the frozen experiment profile and resolve its project-local inputs."""
    path = Path(config_path or DEFAULT_CONFIG).resolve()
    profile = fields.read_json(path)
    for key in ('map_config', 'provenance', 'output_dir', 'reference_field_metrics',
                'reference_table4', 'pattern_statistics', 'comparison_config'):
        profile[key] = str((path.parent / profile[key]).resolve())
    for method in profile['methods'].values():
        method['snapshot'] = str((path.parent / method['snapshot']).resolve())
    return profile


def verify_inputs(profile):
    """Verify migrated input hashes before reproducing the archived experiment."""
    provenance = fields.read_json(profile['provenance'])
    for name, record in provenance['inputs'].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or fields.file_hash(path) != record['sha256']:
            raise ValueError(f'Frozen paper input is missing or changed: {name}')
    return provenance


def pattern_summary(config_path=None):
    """Validate and summarize the 1,656 archived patterns without querying a database."""
    profile = load_profile(config_path)
    provenance = fields.read_json(profile['provenance'])
    source = str(Path(profile['pattern_statistics']).relative_to(PROJECT_ROOT)).replace('\\', '/')
    if fields.file_hash(profile['pattern_statistics']) != provenance['inputs'][source]['sha256']:
        raise ValueError('The archived pattern statistics have changed.')
    patterns = fields.read_json(profile['pattern_statistics'])
    total = sum(row['pattern_count'] for row in patterns)
    if len(patterns) != profile['expected_pattern_groups'] or total != profile['expected_patterns']:
        raise ValueError('Pattern statistics do not match the archived paper experiment.')
    return dict(pattern_groups=len(patterns), patterns=total,
                source_database=profile['pattern_source_database'],
                source_file=str(Path(profile['pattern_statistics']).relative_to(PROJECT_ROOT)),
                sha256=fields.file_hash(profile['pattern_statistics']),
                database_snapshot_included=False,
                provenance_note='Database version identified by the original generalization script; '
                                'the frozen statistics do not replace a database dump.')


def reproduce_method(name, method, profile, output, no_plots=False):
    """Rebuild one method from its archived scores, factor associations, and geometries."""
    snapshot = fields.read_json(method['snapshot'])
    association_count = sum(len(events) for events in snapshot['factor_to_disaster'].values())
    expected = profile['expected_factor_associations'].get(name)
    if expected is not None and association_count != expected:
        raise ValueError(f'{name}: expected {expected} factor associations, found {association_count}.')
    signature = dict(method=name, snapshot_sha256=fields.file_hash(method['snapshot']),
                     map_config_sha256=fields.file_hash(profile['map_config']), render_png=not no_plots)
    manifest_path = output / 'manifest.json'
    if output.exists() and any(output.iterdir()):
        manifest = fields.read_json(manifest_path) if manifest_path.exists() else {}
        status_path = output / 'run_status.json'
        status = fields.read_json(status_path) if status_path.exists() else {}
        outputs = manifest.get('outputs', [])
        files_present = outputs and all(
            Path(item['geotiff']).is_file() and
            (no_plots or (item.get('image') and Path(item['image']).is_file()))
            for item in outputs)
        if manifest.get('paper_reproduction') != signature or status.get('status') != 'complete' or not files_present:
            raise FileExistsError(f'Existing maps do not match the frozen paper inputs: {output}')
        print(f'Reusing verified {name} maps: {output}', flush=True)
        return manifest
    manifest = fields.run(profile['map_config'], snapshot_path=method['snapshot'],
                          output_dir=output, no_plots=no_plots, database=method['database'])
    manifest['paper_reproduction'] = signature
    manifest['factor_association_count'] = association_count
    fields.write_json(manifest_path, manifest)
    return manifest


def verify_field_metrics(results, reference_path):
    """Check all Table 6 and Table 7 values at the precision reported in the paper."""
    with Path(reference_path).open(encoding='utf-8-sig', newline='') as stream:
        reference = list(csv.DictReader(stream))
    key = lambda row: (row['run'], row['kind'], row['field'], canonical_point_set(row['point_set']))
    actual = {key(row): row for row in results}
    metrics = ('AUC_PA', 'HR_at_10pct', 'Lift_at_10pct', 'Area_at_80pct_hit')
    checked = 0
    for expected in reference:
        row = actual.get(key(expected))
        if row is None:
            raise ValueError(f'Missing paper evaluation row: {key(expected)}')
        for metric in metrics:
            scale = 100 if metric in ('HR_at_10pct', 'Area_at_80pct_hit') else 1
            precision = 4 if metric == 'AUC_PA' else 2
            if round(float(row[metric]) * scale, precision) != round(float(expected[metric]) * scale, precision):
                raise ValueError(f'Paper metric differs: {key(expected)}, {metric}')
            checked += 1
    if len(actual) != len(reference):
        raise ValueError('The evaluated fields differ from the paper field selection.')
    return checked


def verify_table4(rows, reference_path):
    """Check all 24 extraction F1 values at the paper's three-decimal precision."""
    with Path(reference_path).open(encoding='utf-8-sig', newline='') as stream:
        reference = list(csv.DictReader(stream))
    actual = {(int(row['shot']), row['method']): row for row in rows}
    metrics = ('geo_micro_f1', 'event_micro_f1', 'event_relation_f1')
    for expected in reference:
        row = actual[(int(expected['shot']), expected['method'])]
        for metric in metrics:
            if round(float(row[metric]), 3) != round(float(expected[metric]), 3):
                raise ValueError(f'Table 4 differs: {expected["shot"]}, {expected["method"]}, {metric}')
    return len(reference) * len(metrics)


def reproduce(config_path=None, output_dir=None, no_plots=False, overwrite=False):
    """Reproduce Tables 4, 6, and 7 and the archived pattern statistics without services."""
    from main.main_extraction_validation import run_table4
    profile = load_profile(config_path)
    provenance = verify_inputs(profile)
    output = Path(output_dir or profile['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields.write_json(output / 'reproduction_status.json', {'status': 'running'})
    try:
        patterns = pattern_summary(config_path)
        fields.write_json(output / 'pattern_statistics_summary.json', patterns)
        table4 = run_table4(output_path=output / 'table4.csv')
        extraction_values = verify_table4(table4, profile['reference_table4'])
        manifests = {}
        runs = []
        for name, method in profile['methods'].items():
            method_output = output / name.lower()
            manifests[name] = reproduce_method(name, method, profile, method_output, no_plots)
            runs.append(dict(name=name,
                             event_dir=str(method_output / 'output_event_maps_grid/geotiff'),
                             factor_dir=str(method_output / 'output_factor_maps_grid_geotiff')))
        comparison = fields.read_json(profile['comparison_config'])
        comparison.update(runs=runs, output_dir=str(output / 'comparison'))
        comparison['factor_fields_by_point_set'] = selected_factor_fields(comparison)
        results = evaluate_fields(Path(profile['comparison_config']), overwrite=overwrite,
                                  config_overrides=comparison)
        make_comparison(results, comparison, output / 'comparison')
        field_values = verify_field_metrics(results, profile['reference_field_metrics'])
        report = dict(status='complete', extraction_values_verified=extraction_values,
                      field_values_verified=field_values, patterns=patterns,
                      input_files_verified=len(provenance['inputs']),
                      source_databases={name: method['database'] for name, method in profile['methods'].items()},
                      factor_associations={name: manifest.get('factor_association_count') for name, manifest in manifests.items()},
                      database_writes=False,
                      scope='Replays archived scores and associations; does not rebuild historical graphs.')
        fields.write_json(output / 'reproduction_status.json', report)
        print(f'Paper reproduction verified: {extraction_values} extraction and {field_values} spatial metric values.', flush=True)
        return report
    except Exception as error:
        fields.write_json(output / 'reproduction_status.json', {'status': 'failed', 'error_type': type(error).__name__})
        raise
