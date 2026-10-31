import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'instantiation_validation'))


def run_oelg(config, maps_only=False, no_plots=False, output_dir=None):
    """Instantiate the OELG baseline or render its existing instance graph."""
    import instantiation_oelg
    arguments = ['--config', str(config)]
    if maps_only:
        arguments.append('--maps-only')
    if no_plots:
        arguments.append('--no-plots')
    if output_dir:
        arguments.extend(['--map-output-dir' if maps_only else '--output-folder', str(output_dir)])
    return instantiation_oelg.main(arguments)


def run_comparison(config, overwrite=False):
    """Evaluate and compare the configured instance importance fields."""
    import compare_importance_fields
    arguments = ['--config', str(config)]
    if overwrite:
        arguments.append('--overwrite')
    return compare_importance_fields.main(arguments)


def main():
    """Run an OELG baseline, draw its maps, or compare existing maps."""
    parser = argparse.ArgumentParser(description='Instance graph instantiation_validation and method comparison')
    parser.add_argument('-t', '--task', required=True, choices=['instantiate', 'map', 'compare'])
    parser.add_argument('--config', type=Path, help='Configuration file for the selected task')
    parser.add_argument('--no-plots', action='store_true', help='Skip PNG generation')
    parser.add_argument('--output-dir', type=Path, help='OELG audit output or map output directory')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing comparison outputs')
    args = parser.parse_args()
    if args.task == 'compare':
        if args.no_plots or args.output_dir:
            parser.error('--no-plots and --output-dir are only available for OELG tasks')
        run_comparison(args.config or PROJECT_ROOT / 'instantiation_validation' / 'importance_comparison_config.json', args.overwrite)
    else:
        if args.overwrite:
            parser.error('--overwrite is only available for comparison')
        run_oelg(args.config or PROJECT_ROOT / 'instantiation_validation' / 'oelg_config.json',
                 maps_only=args.task == 'map', no_plots=args.no_plots, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
