import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def run_evaluation(kind, input_path):
    """Evaluate a prepared extraction workbook."""
    if kind == 'event':
        from extract_validation.validation import event_validata_main
        return event_validata_main(input_path)
    if kind == 'geo':
        from extract_validation.validation import geo_validata_main
        return geo_validata_main(input_path)
    from extract_validation.validation_plot_er import calculate_combined_metrics
    precision, recall, f1 = calculate_combined_metrics(input_path)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    return precision, recall, f1


def prepare_dataset(kind, input_path, output_path, only_raw=False):
    """Prepare a new workbook without modifying the input dataset."""
    from extract_validation.validation import (dataset_process_event_main,
        dataset_process_geo_main, dataset_process_event_relation_main)
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination or destination.exists():
        raise FileExistsError(f'Choose a new output workbook: {destination}')
    if only_raw and kind == 'geo':
        raise ValueError('--only-raw is supported for event and relation datasets')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if kind == 'event':
        dataset_process_event_main(destination, only_raw)
    elif kind == 'geo':
        dataset_process_geo_main(destination)
    else:
        dataset_process_event_relation_main(destination, only_raw)
    print(f'Prepared workbook: {destination}')


def run_plots(kind, base_dir, output_dir, shots):
    """Plot method comparisons from prepared few-shot workbooks."""
    if kind == 'event':
        from extract_validation.validation_plot_e import plot_comparison_results
        return plot_comparison_results(shots, base_dir, output_dir=output_dir)
    if kind == 'geo':
        from extract_validation.validation_plot_geo import plot_geo_comparison_results
        return plot_geo_comparison_results(shots, base_dir, output_dir=output_dir)
    from extract_validation.validation_plot_er import print_performance_table, plot_combined_relation_comparison
    print_performance_table(shots, base_dir)
    return plot_combined_relation_comparison(shots, base_dir, output_dir=output_dir)


def main():
    """Route extraction validation tasks from the command line."""
    parser = argparse.ArgumentParser(description='Prepare, evaluate, or plot extraction validation data')
    parser.add_argument('-t', '--task', required=True, choices=['prepare', 'evaluate', 'plot'])
    parser.add_argument('-k', '--kind', required=True, choices=['event', 'geo', 'relation'])
    parser.add_argument('--input', type=Path, help='Input workbook for prepare or evaluate')
    parser.add_argument('--output', type=Path, help='New output workbook for prepare')
    parser.add_argument('--only-raw', action='store_true', help='Prepare predictions without ground truth')
    parser.add_argument('--base-dir', type=Path, help='Few-shot workbook directory')
    parser.add_argument('--output-dir', type=Path, help='Plot output directory')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 5, 10, 30])
    args = parser.parse_args()
    if args.task in ('prepare', 'evaluate'):
        if args.input is None:
            parser.error('--input is required for prepare and evaluate')
        if args.task == 'prepare':
            if args.output is None:
                parser.error('--output is required for prepare')
            prepare_dataset(args.kind, args.input, args.output, args.only_raw)
        else:
            run_evaluation(args.kind, args.input)
    else:
        base = args.base_dir or PROJECT_ROOT / 'data' / 'extraction_validation_dataset' / ('event_relation' if args.kind == 'relation' else args.kind)
        output = args.output_dir or PROJECT_ROOT / 'data' / 'extraction_validation_dataset' / 'results' / args.kind
        run_plots(args.kind, base, output, args.shots)


if __name__ == '__main__':
    main()
