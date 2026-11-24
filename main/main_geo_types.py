"""Rebuild or validate the Appendix A/B geographic type auxiliary database."""

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.construct_config import configure_from_cli, get_setting, resolve_project_path
from src.other.geo_type_graph import (load_geo_type_rules, open_geo_type_driver, prepare_geo_type_graph,
                                      read_graph, rebuild_geo_type_graph, rules_path, validate_geo_type_graph)


def main():
    """Preview, rebuild, or validate geos-type-1 using shared connection settings."""
    configure_from_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--task', required=True, choices=['preview', 'rebuild', 'validate'])
    parser.add_argument('--config', default=str(project_root / 'construct_config.json'))
    parser.add_argument('--rules', default=str(rules_path()))
    parser.add_argument('-db', '--database', default=get_setting('geocoding').get('type_database', 'geos-type-1'))
    parser.add_argument('--output-dir', default=str(project_root / 'data/geo_type_graph'))
    args = parser.parse_args()
    catalog = load_geo_type_rules(str(resolve_project_path(args.rules)))
    prepared = prepare_geo_type_graph(catalog)
    if args.task == 'preview':
        print(json.dumps(prepared['summary'], ensure_ascii=False, indent=2))
        return 0
    with open_geo_type_driver() as driver:
        if args.task == 'rebuild':
            report = rebuild_geo_type_graph(driver, args.database, prepared, resolve_project_path(args.output_dir))
            print(json.dumps(report['validation'], ensure_ascii=False, indent=2))
        else:
            with driver.session(database=args.database, default_access_mode='READ') as session:
                validation = validate_geo_type_graph(session.execute_read(read_graph), prepared)
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 0 if validation['valid'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
